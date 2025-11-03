import Accelerate
import Foundation
import OSLog

/// A vector database implementation that stores and searches documents using their vector embeddings.
public actor VecturaKit {

    /// Logger for error reporting and warnings
    private static let logger = Logger(
        subsystem: "com.vecturakit",
        category: "VecturaKit"
    )

    /// The configuration for this vector database instance.
    private let config: VecturaConfig

    /// The embedder used to generate vector embeddings from text.
    private let embedder: VecturaEmbedder

    /// The actual dimension of vectors, either from config or auto-detected from embedder.
    private var actualDimension: Int?

    /// In-memory cache of all documents.
    private var documents: [UUID: VecturaDocument]

    /// The storage directory for documents.
    private let storageDirectory: URL

    /// The storage provider that handles document persistence.
    private let storageProvider: VecturaStorage

    /// Optional indexed storage provider for efficient large-scale operations.
    private let indexedStorage: IndexedVecturaStorage?

    /// Indicates whether indexed storage is available for use.
    /// This is determined at initialization based on storage provider capabilities.
    private let indexedModeAvailable: Bool

    /// Cached normalized embeddings for faster searches.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]

    /// BM25 index for text search
    private var bm25Index: BM25Index?

    // MARK: - Initialization

    /// Initializes a new VecturaKit instance with the specified configuration and embedder.
    ///
    /// - Parameters:
    ///   - config: Configuration options for the vector database.
    ///   - embedder: The embedder to use for generating vector embeddings from text.
    ///   - storageProvider: Optional custom storage provider. If nil, uses FileStorageProvider.
    public init(
        config: VecturaConfig,
        embedder: VecturaEmbedder,
        storageProvider: VecturaStorage? = nil
    ) async throws {
        // Validate memory strategy parameters
        switch config.memoryStrategy {
        case .automatic(let threshold, let multiplier, let batch, let maxConcurrent):
            guard threshold >= 0 else {
                throw VecturaError.invalidInput("Automatic threshold must be non-negative, got \(threshold)")
            }
            guard multiplier > 0 else {
                throw VecturaError.invalidInput("candidateMultiplier must be positive, got \(multiplier)")
            }
            guard batch > 0 else {
                throw VecturaError.invalidInput("batchSize must be positive, got \(batch)")
            }
            guard maxConcurrent > 0 else {
                throw VecturaError.invalidInput("maxConcurrentBatches must be positive, got \(maxConcurrent)")
            }
        case .indexed(let multiplier, let batch, let maxConcurrent):
            guard multiplier > 0 else {
                throw VecturaError.invalidInput("candidateMultiplier must be positive, got \(multiplier)")
            }
            guard batch > 0 else {
                throw VecturaError.invalidInput("batchSize must be positive, got \(batch)")
            }
            guard maxConcurrent > 0 else {
                throw VecturaError.invalidInput("maxConcurrentBatches must be positive, got \(maxConcurrent)")
            }
        case .fullMemory:
            break  // No validation needed
        }

        self.config = config
        self.embedder = embedder
        self.documents = [:]

        if let customStorageDirectory = config.directoryURL {
            let databaseDirectory = customStorageDirectory.appending(path: config.name)
            if !FileManager.default.fileExists(atPath: databaseDirectory.path(percentEncoded: false)) {
                try FileManager.default.createDirectory(
                    at: databaseDirectory, withIntermediateDirectories: true)
            }
            self.storageDirectory = databaseDirectory
        } else {
            // Create default storage directory
            guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                throw VecturaError.loadFailed("Could not access document directory")
            }
            self.storageDirectory = documentsURL
                .appendingPathComponent("VecturaKit")
                .appendingPathComponent(config.name)
        }

        // Use custom storage provider if provided, otherwise use FileStorageProvider.
        if let customProvider = storageProvider {
            self.storageProvider = customProvider
        } else {
            // Note: FileStorageProvider creates its storage directory in its initializer.
            self.storageProvider = try FileStorageProvider(storageDirectory: storageDirectory)
        }

        // Check if storage provider supports indexed operations
        self.indexedStorage = self.storageProvider as? IndexedVecturaStorage

        // Determine if indexed mode is available based on storage capabilities and strategy
        self.indexedModeAvailable = Self.isIndexedModeAvailable(
            strategy: config.memoryStrategy,
            indexedStorage: self.indexedStorage
        )

        // Log if indexed mode was requested but fallback occurred
        if case .indexed = config.memoryStrategy, !indexedModeAvailable {
            let msg = "Indexed mode requested but storage provider doesn't support it. Using fullMemory."
            Self.logger.info("\(msg)")
        }

        // Initialize based on memory strategy
        try await initializeWithStrategy()
    }

    // MARK: - Public API

    /// Adds a single document to the vector store.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    /// - Returns: The ID of the added document.
    public func addDocument(text: String, id: UUID? = nil) async throws -> UUID {
        let ids = try await addDocuments(texts: [text], ids: id.map { [$0] })
        return ids[0]
    }

    /// Adds multiple documents to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - ids: Optional unique identifiers for the documents.
    /// - Returns: The IDs of the added documents.
    public func addDocuments(texts: [String], ids: [UUID]? = nil) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }

        // Get embeddings from the embedder
        let embeddings = try await embedder.embed(texts: texts)

        // Detect dimension from first embedding
        if actualDimension == nil {
            actualDimension = try await embedder.dimension
        }

        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Could not determine embedder dimension")
        }

        // Validate dimension if specified in config
        if let configDimension = config.dimension, configDimension != dimension {
            throw VecturaError.dimensionMismatch(expected: configDimension, got: dimension)
        }

        // Validate embeddings
        for embedding in embeddings {
            try validateDimension(embedding)
        }

        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()

        for i in 0..<texts.count {
            let docId = ids?[i] ?? UUID()
            let doc = VecturaDocument(
                id: docId,
                text: texts[i],
                embedding: embeddings[i]
            )
            documentsToSave.append(doc)
            documentIds.append(docId)
        }

        // Save documents to storage first to ensure atomicity
        let storage = self.storageProvider
        try await withThrowingTaskGroup(of: Void.self) { group in
            for doc in documentsToSave {
                group.addTask {
                    try await storage.saveDocument(doc)
                }
            }

            try await group.waitForAll()
        }

        // Only update in-memory state after successful persistence
        for doc in documentsToSave {
            let normalized = try normalizeEmbedding(doc.embedding)
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc

            // Incrementally update BM25 index
            if var index = bm25Index {
                // Check if document with this ID already exists in BM25 index
                if index.containsDocument(withID: doc.id) {
                    // Update existing document to keep index in sync
                    index.updateDocument(doc)
                } else {
                    // Add new document
                    index.addDocument(doc)
                }
                bm25Index = index
            } else {
                // Initialize index if it doesn't exist
                let allDocs = Array(documents.values)
                bm25Index = BM25Index(
                    documents: allDocs,
                    k1: config.searchOptions.k1,
                    b: config.searchOptions.b
                )
            }
        }

        return documentIds
    }

    /// Searches for similar documents using a pre-computed query embedding.
    ///
    /// - Parameters:
    ///   - query: The query vector to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by similarity.
    public func search(
        query queryEmbedding: [Float],
        numResults: Int? = nil,
        threshold: Float? = nil
    ) async throws -> [VecturaSearchResult] {
        // Detect dimension if not yet set
        if actualDimension == nil {
            actualDimension = try await embedder.dimension
        }

        guard actualDimension != nil else {
            throw VecturaError.invalidInput("Could not determine embedder dimension")
        }

        // Validate query embedding dimension
        try validateDimension(queryEmbedding)

        // Dynamically determine which search strategy to use
        let shouldUseIndexed = try await shouldUseIndexedMode()

        if shouldUseIndexed {
            return try await searchWithIndex(
                query: queryEmbedding,
                numResults: numResults,
                threshold: threshold
            )
        } else {
            return try await searchInMemory(
                query: queryEmbedding,
                numResults: numResults,
                threshold: threshold
            )
        }
    }

    /// Searches for similar documents using a text query with hybrid search (vector + BM25).
    ///
    /// Note: Hybrid search is only available in fullMemory mode. In indexed mode,
    /// only vector search is performed to avoid loading all documents into memory.
    ///
    /// - Parameters:
    ///   - query: The text query to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by hybrid score (fullMemory) or vector score (indexed).
    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil
    ) async throws -> [VecturaSearchResult] {
        // Detect dimension if not yet set
        if actualDimension == nil {
            actualDimension = try await embedder.dimension
        }

        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Could not determine embedder dimension")
        }

        // Validate that embedder dimension matches config dimension if already set
        if let configDimension = config.dimension, configDimension != dimension {
            throw VecturaError.dimensionMismatch(expected: configDimension, got: dimension)
        }

        // Get vector embedding and perform vector search
        let queryEmbedding = try await embedder.embed(text: query)

        // Validate dimension
        if queryEmbedding.count != dimension {
            throw VecturaError.dimensionMismatch(expected: dimension, got: queryEmbedding.count)
        }

        // Get all vector results first (no filtering) for potential hybrid scoring
        let vectorResults = try await search(
            query: queryEmbedding,
            numResults: nil,
            threshold: nil
        )

        // Check if we should use hybrid search
        // Hybrid search is only supported in fullMemory mode to avoid loading all documents
        let useIndexed = try await shouldUseIndexedMode()
        if useIndexed {
            // Indexed mode: return vector-only results with filtering
            var results = vectorResults

            if let threshold = threshold ?? config.searchOptions.minThreshold {
                results = results.filter { $0.score >= threshold }
            }

            let limit = numResults ?? config.searchOptions.defaultNumResults
            return Array(results.prefix(limit))
        }

        // FullMemory mode: perform hybrid search with BM25

        // Initialize BM25 index if needed
        if bm25Index == nil {
            let docs = documents.values.map { $0 }
            bm25Index = BM25Index(
                documents: docs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }

        // Get BM25 text search results
        let requestedLimit = numResults ?? config.searchOptions.defaultNumResults
        let bm25Limit = min(requestedLimit * 2, documents.count)
        let bm25Results = bm25Index?.search(
            query: query,
            topK: bm25Limit
        ) ?? []

        // Create a map of document IDs to their BM25 scores
        let bm25Scores = Dictionary(
            bm25Results.map { ($0.document.id, $0.score) },
            uniquingKeysWith: { first, _ in first }
        )

        // Combine scores using hybrid scoring
        var hybridResults = vectorResults.map { result in
            let bm25Score = bm25Scores[result.id] ?? 0
            let hybridScore = VecturaDocument.calculateHybridScore(
                vectorScore: result.score,
                bm25Score: bm25Score,
                weight: config.searchOptions.hybridWeight,
                normalizationFactor: config.searchOptions.bm25NormalizationFactor
            )

            return VecturaSearchResult(
                id: result.id,
                text: result.text,
                score: hybridScore,
                createdAt: result.createdAt
            )
        }

        hybridResults.sort { $0.score > $1.score }

        if let threshold = threshold ?? config.searchOptions.minThreshold {
            hybridResults = hybridResults.filter { $0.score >= threshold }
        }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(hybridResults.prefix(limit))
    }

    /// Removes all documents from the vector store.
    public func reset() async throws {
        // Get all document IDs from storage, not just from memory cache
        // This ensures we delete persisted documents even in indexed mode
        let allDocs = try await storageProvider.loadDocuments()
        let documentIds = allDocs.map { $0.id }
        try await deleteDocuments(ids: documentIds)
    }

    /// Deletes specific documents from the vector store.
    ///
    /// - Parameter ids: The IDs of documents to delete.
    public func deleteDocuments(ids: [UUID]) async throws {
        // Incrementally remove documents from BM25 index
        if var index = bm25Index {
            for id in ids {
                index.removeDocument(id)
            }
            bm25Index = index
        }

        for id in ids {
            documents[id] = nil
            normalizedEmbeddings[id] = nil

            // Delete using storage provider
            try await storageProvider.deleteDocument(withID: id)
        }
    }

    /// Updates an existing document with new text.
    ///
    /// - Parameters:
    ///   - id: The ID of the document to update.
    ///   - newText: The new text content for the document.
    public func updateDocument(id: UUID, newText: String) async throws {
        // Try to get from cache first, otherwise load from storage
        let oldDocument: VecturaDocument
        if let cached = documents[id] {
            oldDocument = cached
        } else {
            // In indexed mode, document may not be in memory - load it from storage
            if let indexed = indexedStorage {
                let loaded = try await indexed.loadDocuments(ids: [id])
                guard let doc = loaded[id] else {
                    throw VecturaError.documentNotFound(id)
                }
                oldDocument = doc
            } else {
                // Fall back to loading all documents and finding the target
                let allDocs = try await storageProvider.loadDocuments()
                guard let doc = allDocs.first(where: { $0.id == id }) else {
                    throw VecturaError.documentNotFound(id)
                }
                oldDocument = doc
            }
        }

        // Generate new embedding
        let newEmbedding = try await embedder.embed(text: newText)

        // Detect dimension if not yet set
        if actualDimension == nil {
            actualDimension = try await embedder.dimension
        }

        // Validate dimension
        try validateDimension(newEmbedding)

        // Create updated document, preserving original creation date
        let updatedDoc = VecturaDocument(
            id: id,
            text: newText,
            embedding: newEmbedding,
            createdAt: oldDocument.createdAt
        )

        // 1. Persist the updated document first to ensure atomicity
        try await storageProvider.saveDocument(updatedDoc)

        // 2. Only update in-memory state after successful persistence
        documents[id] = updatedDoc

        let normalized = try normalizeEmbedding(updatedDoc.embedding)
        normalizedEmbeddings[id] = normalized

        // 3. Incrementally update BM25 index
        if var index = bm25Index {
            index.updateDocument(updatedDoc)
            bm25Index = index
        } else {
            // Initialize index if it doesn't exist
            let allDocs = Array(documents.values)
            bm25Index = BM25Index(
                documents: allDocs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }
    }

    // MARK: - Public Properties

    /// Returns the number of documents currently stored in the vector database.
    public var documentCount: Int {
        return documents.count
    }

    /// Returns all documents currently stored in the vector database.
    /// - Returns: Array of VecturaDocument objects
    public func getAllDocuments() -> [VecturaDocument] {
        return Array(documents.values)
    }

    // MARK: - Private

    private func l2Norm(_ v: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
        return sqrt(sumSquares)
    }

    /// Normalizes a vector using L2 normalization
    /// - Parameter embedding: The vector to normalize
    /// - Returns: The normalized vector
    /// - Throws: VecturaError.invalidInput if the vector has zero norm
    private func normalizeEmbedding(_ embedding: [Float]) throws -> [Float] {
        let norm = l2Norm(embedding)

        // Check for zero-norm vectors which cannot be normalized
        if norm < 1e-10 {
            throw VecturaError.invalidInput("Cannot normalize zero-norm embedding vector")
        }

        var divisor = norm
        var normalized = [Float](repeating: 0, count: embedding.count)
        vDSP_vsdiv(embedding, 1, &divisor, &normalized, 1, vDSP_Length(embedding.count))
        return normalized
    }

    /// Validates that the embedding dimension matches the expected dimension
    private func validateDimension(_ embedding: [Float]) throws {
        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Dimension not yet determined")
        }

        if embedding.count != dimension {
            throw VecturaError.dimensionMismatch(expected: dimension, got: embedding.count)
        }

        // Also validate against config dimension if specified
        if let configDimension = config.dimension, configDimension != dimension {
            throw VecturaError.dimensionMismatch(expected: configDimension, got: dimension)
        }
    }

    // MARK: - Initialization Strategies

    /// Determines whether indexed mode is available based on strategy and storage capabilities.
    ///
    /// - Parameters:
    ///   - strategy: The configured memory strategy
    ///   - indexedStorage: Optional indexed storage provider
    /// - Returns: True if indexed mode is available, false otherwise
    private static func isIndexedModeAvailable(
        strategy: VecturaConfig.MemoryStrategy,
        indexedStorage: IndexedVecturaStorage?
    ) -> Bool {
        // If no indexed storage is available, indexed mode cannot be used
        guard indexedStorage != nil else {
            return false
        }

        // Check if strategy allows indexed mode
        switch strategy {
        case .automatic, .indexed:
            return true
        case .fullMemory:
            return false
        }
    }

    /// Initializes VecturaKit based on the configured memory strategy.
    private func initializeWithStrategy() async throws {
        // Determine if we should start with indexed mode
        let useIndexed = try await shouldUseIndexedMode()

        // If not using indexed mode, load all documents upfront
        if !useIndexed {
            try await loadAllDocuments()
        }
        // Otherwise, documents will be loaded on-demand during search
    }

    /// Loads all documents into memory (backward-compatible behavior).
    private func loadAllDocuments() async throws {
        let storedDocuments = try await self.storageProvider.loadDocuments()
        for doc in storedDocuments {
            self.documents[doc.id] = doc
            // Compute normalized embedding and store in cache
            let normalized = try normalizeEmbedding(doc.embedding)
            self.normalizedEmbeddings[doc.id] = normalized
        }
    }

    /// Gets the total document count efficiently.
    private func getTotalDocumentCount() async throws -> Int {
        if let indexed = indexedStorage {
            do {
                return try await indexed.getTotalDocumentCount()
            } catch {
                let errMsg = "Failed to retrieve document count from indexed storage: \(error.localizedDescription)"
                throw VecturaError.loadFailed(errMsg)
            }
        } else {
            // Fallback: load all documents to count them
            do {
                return try await storageProvider.loadDocuments().count
            } catch {
                throw VecturaError.loadFailed("Failed to load documents for counting: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Search Strategies

    /// Determines whether to use indexed search mode.
    ///
    /// For automatic strategy, this dynamically evaluates based on current document count.
    /// For other strategies, returns a fixed decision based on strategy and availability.
    private func shouldUseIndexedMode() async throws -> Bool {
        // If indexed storage is not available, cannot use indexed mode
        guard indexedModeAvailable else {
            return false
        }

        switch config.memoryStrategy {
        case .automatic(let threshold, _, _, _):
            // Re-evaluate based on current document count
            let currentCount = try await getTotalDocumentCount()
            return currentCount >= threshold
        case .indexed:
            return true
        case .fullMemory:
            return false
        }
    }

    /// In-memory vector search (original implementation).
    private func searchInMemory(
        query queryEmbedding: [Float],
        numResults: Int?,
        threshold: Float?
    ) async throws -> [VecturaSearchResult] {
        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Model dimension not detected")
        }

        // Normalize the query vector
        let normalizedQuery = try normalizeEmbedding(queryEmbedding)

        // Build a matrix of normalized document embeddings in row-major order
        var docIds = [UUID]()
        var matrix = [Float]()
        matrix.reserveCapacity(documents.count * dimension)

        for doc in documents.values {
            if let normalized = normalizedEmbeddings[doc.id] {
                docIds.append(doc.id)
                matrix.append(contentsOf: normalized)
            }
        }

        let docsCount = docIds.count
        if docsCount == 0 {
            return []
        }

        var similarities = [Float](repeating: 0, count: docsCount)

        // Compute all similarities using matrix-vector multiplication
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            docsCount,
            dimension,
            1.0,
            matrix,
            dimension,
            normalizedQuery,
            1,
            0.0,
            &similarities,
            1
        )

        // Construct results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(docsCount)

        for (i, similarity) in similarities.enumerated() {
            if let minT = threshold ?? config.searchOptions.minThreshold, similarity < minT {
                continue
            }
            if let doc = documents[docIds[i]] {
                results.append(
                    VecturaSearchResult(
                        id: doc.id,
                        text: doc.text,
                        score: similarity,
                        createdAt: doc.createdAt
                    )
                )
            }
        }

        results.sort { $0.score > $1.score }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
    }

    // MARK: - Indexed Search Parameters

    /// Stores indexed search configuration parameters.
    private struct IndexedSearchParams {
        let candidateMultiplier: Int
        let batchSize: Int
        let maxConcurrentBatches: Int
    }

    /// Extracts indexed search parameters from current strategy configuration.
    private func extractIndexedSearchParams() -> IndexedSearchParams {
        switch config.memoryStrategy {
        case .indexed(let multiplier, let batch, let maxConcurrent):
            return IndexedSearchParams(
                candidateMultiplier: multiplier,
                batchSize: batch,
                maxConcurrentBatches: maxConcurrent
            )
        case .automatic(_, let multiplier, let batch, let maxConcurrent):
            return IndexedSearchParams(
                candidateMultiplier: multiplier,
                batchSize: batch,
                maxConcurrentBatches: maxConcurrent
            )
        case .fullMemory:
            assertionFailure("searchWithIndex should not be called with fullMemory strategy")
            return IndexedSearchParams(
                candidateMultiplier: VecturaConfig.MemoryStrategy.defaultCandidateMultiplier,
                batchSize: VecturaConfig.MemoryStrategy.defaultBatchSize,
                maxConcurrentBatches: VecturaConfig.MemoryStrategy.defaultMaxConcurrentBatches
            )
        }
    }

    /// Indexed vector search using storage-layer filtering.
    private func searchWithIndex(
        query queryEmbedding: [Float],
        numResults: Int?,
        threshold: Float?
    ) async throws -> [VecturaSearchResult] {
        guard let indexedStorage = self.indexedStorage else {
            // Fallback to in-memory search
            return try await searchInMemory(
                query: queryEmbedding,
                numResults: numResults,
                threshold: threshold
            )
        }

        guard actualDimension != nil else {
            throw VecturaError.invalidInput("Model dimension not detected")
        }

        let topK = numResults ?? config.searchOptions.defaultNumResults
        let params = extractIndexedSearchParams()
        let prefilterSize = topK * params.candidateMultiplier

        // Stage 1: Get candidate document IDs from storage layer
        let candidateIds = try await indexedStorage.searchCandidates(
            queryEmbedding: queryEmbedding,
            topK: topK,
            prefilterSize: prefilterSize
        )

        if candidateIds.isEmpty {
            return []
        }

        // Stage 2: Load candidate documents in batches with error handling
        let candidates = try await loadDocumentsBatched(
            ids: candidateIds,
            batchSize: params.batchSize,
            maxConcurrentBatches: params.maxConcurrentBatches,
            storage: indexedStorage
        )

        if candidates.isEmpty {
            return []
        }

        // Stage 3: Compute exact similarities for candidates
        let normalizedQuery = try normalizeEmbedding(queryEmbedding)

        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Model dimension not detected")
        }

        // Build a matrix of normalized candidate embeddings
        var candidateDocIds = [UUID]()
        var candidateDocs = [VecturaDocument]()
        var matrix = [Float]()
        matrix.reserveCapacity(candidates.count * dimension)

        for (id, doc) in candidates {
            let normalizedDoc = try normalizeEmbedding(doc.embedding)
            candidateDocIds.append(id)
            candidateDocs.append(doc)
            matrix.append(contentsOf: normalizedDoc)
        }

        let candidatesCount = candidateDocIds.count
        if candidatesCount == 0 {
            return []
        }

        let M = Int32(candidatesCount)
        let N = Int32(dimension)
        var similarities = [Float](repeating: 0, count: candidatesCount)

        let mInt = Int(M)
        let nInt = Int(N)
        let ldInt = Int(N)

        // Compute similarities using matrix-vector multiplication
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            mInt,
            nInt,
            1.0,
            matrix,
            ldInt,
            normalizedQuery,
            1,
            0.0,
            &similarities,
            1
        )

        // Construct results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(candidatesCount)

        for (i, similarity) in similarities.enumerated() {
            if let minT = threshold ?? config.searchOptions.minThreshold, similarity < minT {
                continue
            }
            let doc = candidateDocs[i]
            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: similarity,
                    createdAt: doc.createdAt
                )
            )
        }

        // Sort by score and return top K
        results.sort { $0.score > $1.score }
        return Array(results.prefix(topK))
    }

    // MARK: - Batch Loading with Error Handling

    /// Information about a failed batch load operation.
    private struct BatchLoadFailure: Error {
        let batchIndex: Int
        let error: Error
        let affectedIds: [UUID]
    }

    /// Loads documents in batches with controlled concurrency and error handling.
    ///
    /// This method implements a robust batch loading strategy:
    /// 1. Splits document IDs into batches for parallel processing
    /// 2. Limits concurrent operations to prevent resource exhaustion
    /// 3. Handles partial failures gracefully (failed batches are logged but don't fail the entire operation)
    /// 4. Uses streaming aggregation to avoid large memory allocations
    ///
    /// - Parameters:
    ///   - ids: Array of document IDs to load
    ///   - batchSize: Number of documents per batch (default: 100)
    ///   - maxConcurrentBatches: Maximum number of concurrent batch operations (default: 4)
    ///   - storage: The indexed storage provider to use for loading
    /// - Returns: Dictionary mapping document IDs to their documents (may not include all requested
    ///   IDs if some batches failed)
    /// - Throws: Only if ALL batches fail; otherwise returns partial results
    private func loadDocumentsBatched(
        ids: [UUID],
        batchSize: Int,
        maxConcurrentBatches: Int,
        storage: IndexedVecturaStorage
    ) async throws -> [UUID: VecturaDocument] {
        // For small candidate sets, load directly without batching overhead
        guard ids.count > batchSize else {
            return try await storage.loadDocuments(ids: ids)
        }

        // Split IDs into batches
        let batches = stride(from: 0, to: ids.count, by: batchSize).map { startIndex -> (index: Int, ids: [UUID]) in
            let endIndex = min(startIndex + batchSize, ids.count)
            return (index: startIndex / batchSize, ids: Array(ids[startIndex..<endIndex]))
        }

        // Load batches with controlled concurrency
        var allDocuments: [UUID: VecturaDocument] = [:]
        var failures: [BatchLoadFailure] = []

        // Process batches with concurrency limit
        typealias BatchResult = Result<(Int, [UUID: VecturaDocument]), BatchLoadFailure>
        for batchGroup in batches.chunked(into: maxConcurrentBatches) {
            let groupResults = await withTaskGroup(of: BatchResult.self) { group in
                for batch in batchGroup {
                    group.addTask {
                        do {
                            let docs = try await storage.loadDocuments(ids: batch.ids)
                            return .success((batch.index, docs))
                        } catch {
                            return .failure(BatchLoadFailure(
                                batchIndex: batch.index,
                                error: error,
                                affectedIds: batch.ids
                            ))
                        }
                    }
                }

                var results: [Result<(Int, [UUID: VecturaDocument]), BatchLoadFailure>] = []
                for await result in group {
                    results.append(result)
                }
                return results
            }

            // Process results: merge successes, collect failures
            for result in groupResults {
                switch result {
                case .success(let (_, documents)):
                    // Stream aggregation: merge documents incrementally
                    allDocuments.merge(documents) { _, new in new }
                case .failure(let failure):
                    failures.append(failure)
                }
            }
        }

        // Log failures if any (without failing the entire operation)
        if !failures.isEmpty {
            let totalFailedIds = failures.reduce(0) { $0 + $1.affectedIds.count }
            Self.logger.warning(
                "Warning: \(failures.count) batch(es) failed to load, \(totalFailedIds) documents unavailable"
            )
            for failure in failures {
                Self.logger.warning(
                    "  - Batch \(failure.batchIndex): \(failure.error.localizedDescription)"
                )
            }
        }

        // Only throw if we got NO results at all
        if allDocuments.isEmpty && !failures.isEmpty {
            let firstErrorMsg = failures.first?.error.localizedDescription ?? "unknown"
            let errMsg = "Failed to load any candidate documents. " +
                "All \(failures.count) batch(es) failed. First error: \(firstErrorMsg)"
            throw VecturaError.loadFailed(errMsg)
        }

        return allDocuments
    }
}
