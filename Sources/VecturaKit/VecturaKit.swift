import Accelerate
import Foundation

/// A vector database implementation that stores and searches documents using their vector embeddings.
public actor VecturaKit {

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
            self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first!
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

        // Load existing documents using the storage provider.
        let storedDocuments = try await self.storageProvider.loadDocuments()
        for doc in storedDocuments {
            self.documents[doc.id] = doc
            // Compute normalized embedding and store in cache.
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            self.normalizedEmbeddings[doc.id] = normalized
        }
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
            if embedding.count != dimension {
                throw VecturaError.dimensionMismatch(
                    expected: dimension,
                    got: embedding.count
                )
            }
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

        for doc in documentsToSave {
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
        }

        let allDocs = Array(documents.values)

        bm25Index = BM25Index(
            documents: allDocs,
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )

        // Save documents using the storage provider sequentially to avoid concurrency issues
        for doc in documentsToSave {
            try await storageProvider.saveDocument(doc)
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

        guard let dimension = actualDimension else {
            throw VecturaError.invalidInput("Could not determine embedder dimension")
        }

        if queryEmbedding.count != dimension {
            throw VecturaError.dimensionMismatch(
                expected: dimension,
                got: queryEmbedding.count
            )
        }

        // Normalize the query vector
        let norm = l2Norm(queryEmbedding)
        var divisor = norm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(queryEmbedding, 1, &divisor, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))

        // Build a matrix of normalized document embeddings in row-major order
        var docIds = [UUID]()
        var matrix = [Float]()
        matrix.reserveCapacity(documents.count * dimension)  // Pre-allocate for better performance

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

        let M = Int32(docsCount)  // number of rows (documents)
        let N = Int32(dimension)  // number of columns (embedding dimension)
        var similarities = [Float](repeating: 0, count: docsCount)

        // Convert Int32 to Int for LAPACK compatibility
        let mInt = Int(M)  // Convert number of rows
        let nInt = Int(N)  // Convert number of columns
        let ldInt = Int(N) // Convert leading dimension

        // Compute all similarities at once using matrix-vector multiplication
        // Matrix is in row-major order, so we use CblasNoTrans
        cblas_sgemv(
            CblasRowMajor,    // matrix layout
            CblasNoTrans,     // no transpose needed for row-major
            mInt,             // number of rows (documents) as Int
            nInt,             // number of columns (dimension) as Int
            1.0,              // alpha scaling factor
            matrix,           // matrix
            ldInt,            // leading dimension as Int
            normalizedQuery,  // vector
            1,                // vector increment
            0.0,              // beta scaling factor
            &similarities,    // result vector
            1                 // result increment
        )

        // Construct the results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(docsCount)  // Pre-allocate for better performance

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

    /// Searches for similar documents using a text query with hybrid search (vector + BM25).
    ///
    /// - Parameters:
    ///   - query: The text query to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by hybrid score.
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

        // Initialize BM25 index if needed
        if bm25Index == nil {
            let docs = documents.values.map { $0 }
            bm25Index = BM25Index(
                documents: docs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }

        // Get vector similarity results
        let queryEmbedding = try await embedder.embed(text: query)

        // Validate dimension
        if queryEmbedding.count != dimension {
            throw VecturaError.dimensionMismatch(expected: dimension, got: queryEmbedding.count)
        }

        let vectorResults = try await search(
            query: queryEmbedding,
            numResults: nil,
            threshold: nil
        )

        let bm25Results = bm25Index?.search(
            query: query,
            topK: documents.count
        ) ?? []

        // Create a map of document IDs to their BM25 scores
        let bm25Scores = Dictionary(
            bm25Results.map { ($0.document.id, $0.score) },
            uniquingKeysWith: { first, _ in first }
        )

        // Combine scores using hybrid scoring
        var hybridResults = vectorResults.map { result in
            let bm25Score = bm25Scores[result.id] ?? 0
            let hybridScore = VecturaDocument(
                id: result.id,
                text: result.text,
                embedding: []
            ).hybridScore(
                vectorScore: result.score,
                bm25Score: bm25Score,
                weight: config.searchOptions.hybridWeight
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
        let documentIds = Array(documents.keys)
        try await deleteDocuments(ids: documentIds)
    }

    /// Deletes specific documents from the vector store.
    ///
    /// - Parameter ids: The IDs of documents to delete.
    public func deleteDocuments(ids: [UUID]) async throws {
        if bm25Index != nil {
            let remainingDocs = documents.values.filter { !ids.contains($0.id) }
            bm25Index = BM25Index(
                documents: Array(remainingDocs),
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
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
        try await deleteDocuments(ids: [id])
        _ = try await addDocument(text: newText, id: id)
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
}
