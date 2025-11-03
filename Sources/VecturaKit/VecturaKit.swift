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

    /// The search engine for executing queries.
    private let searchEngine: VecturaSearchEngine

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

    // MARK: - Initialization

    /// Initializes a new VecturaKit instance with the specified configuration and embedder.
    ///
    /// - Parameters:
    ///   - config: Configuration options for the vector database.
    ///   - embedder: The embedder to use for generating vector embeddings from text.
    ///   - storageProvider: Optional custom storage provider. If nil, uses FileStorageProvider.
    ///   - searchEngine: Optional custom search engine. If nil, uses default based on config.
    public init(
        config: VecturaConfig,
        embedder: VecturaEmbedder,
        storageProvider: VecturaStorage? = nil,
        searchEngine: VecturaSearchEngine? = nil
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
            self.storageProvider = try FileStorageProvider(storageDirectory: storageDirectory)
        }

        // Check if storage provider supports indexed operations
        self.indexedStorage = self.storageProvider as? IndexedVecturaStorage

        // Initialize search engine (use custom or create default)
        if let customEngine = searchEngine {
            self.searchEngine = customEngine
        } else {
            // Create default search engine based on config
            self.searchEngine = Self.createDefaultSearchEngine(
                config: config,
                embedder: embedder
            )
        }

        // Detect dimension
        self.actualDimension = try await embedder.dimension

        // Load documents based on strategy
        try await loadDocuments()
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
            documents[doc.id] = doc

            // Notify search engine to index document
            try await searchEngine.indexDocument(doc)
        }

        return documentIds
    }

    /// Searches for similar documents using a text query.
    ///
    /// - Parameters:
    ///   - query: The text query to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by relevance.
    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil
    ) async throws -> [VecturaSearchResult] {
        let options = SearchOptions(
            numResults: numResults ?? config.searchOptions.defaultNumResults,
            threshold: threshold ?? config.searchOptions.minThreshold
        )

        return try await searchEngine.search(
            query: .text(query),
            context: self,
            options: options
        )
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
        // Validate query embedding dimension
        try validateDimension(queryEmbedding)

        let options = SearchOptions(
            numResults: numResults ?? config.searchOptions.defaultNumResults,
            threshold: threshold ?? config.searchOptions.minThreshold
        )

        return try await searchEngine.search(
            query: .vector(queryEmbedding),
            context: self,
            options: options
        )
    }

    /// Removes all documents from the vector store.
    public func reset() async throws {
        // Get all document IDs from storage
        let allDocs = try await storageProvider.loadDocuments()
        let documentIds = allDocs.map { $0.id }
        try await deleteDocuments(ids: documentIds)
    }

    /// Deletes specific documents from the vector store.
    ///
    /// - Parameter ids: The IDs of documents to delete.
    public func deleteDocuments(ids: [UUID]) async throws {
        for id in ids {
            documents[id] = nil

            // Delete using storage provider
            try await storageProvider.deleteDocument(withID: id)

            // Notify search engine
            try await searchEngine.removeDocument(id: id)
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

        // 3. Notify search engine
        try await searchEngine.removeDocument(id: id)
        try await searchEngine.indexDocument(updatedDoc)
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

    /// Creates default search engine based on configuration
    private static func createDefaultSearchEngine(
        config: VecturaConfig,
        embedder: VecturaEmbedder
    ) -> VecturaSearchEngine {
        // Convert memory strategy to search strategy
        let vectorStrategy: VectorSearchEngine.Strategy
        switch config.memoryStrategy {
        case .fullMemory:
            vectorStrategy = .inMemory
        case .indexed(let multiplier, _, _):
            vectorStrategy = .indexed(candidateMultiplier: multiplier)
        case .automatic(let threshold, _, _, _):
            vectorStrategy = .automatic(threshold: threshold)
        }

        // Create vector search engine
        let vectorEngine = VectorSearchEngine(
            embedder: embedder,
            strategy: vectorStrategy
        )

        // For backward compatibility, wrap in hybrid search with BM25
        let bm25Engine = BM25SearchEngine(
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )

        return HybridSearchEngine(
            vectorEngine: vectorEngine,
            textEngine: bm25Engine,
            vectorWeight: config.searchOptions.hybridWeight
        )
    }

    private func loadDocuments() async throws {
        // Always try to load from storage
        let storedDocuments = try await storageProvider.loadDocuments()
        for doc in storedDocuments {
            documents[doc.id] = doc
        }
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

    /// Gets the total document count efficiently.
    private func getTotalDocumentCount() async throws -> Int {
        if let indexed = indexedStorage {
            do {
                return try await indexed.getTotalDocumentCount()
            } catch {
                throw VecturaError.loadFailed("Failed to retrieve document count from indexed storage: \(error.localizedDescription)")
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
}

// MARK: - SearchContext Implementation

extension VecturaKit: SearchContext {
    public func getAllDocuments() async throws -> [VecturaDocument] {
        if !documents.isEmpty {
            return Array(documents.values)
        }
        return try await storageProvider.loadDocuments()
    }

    public func getDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
        // Try from cache first
        var result: [UUID: VecturaDocument] = [:]
        var missingIds: [UUID] = []

        for id in ids {
            if let doc = documents[id] {
                result[id] = doc
            } else {
                missingIds.append(id)
            }
        }

        // Load missing documents from storage
        if !missingIds.isEmpty {
            if let indexedStorage = indexedStorage {
                let loaded = try await indexedStorage.loadDocuments(ids: missingIds)
                result.merge(loaded) { _, new in new }
            } else {
                // Fallback: load all and filter
                let allDocs = try await storageProvider.loadDocuments()
                for doc in allDocs where missingIds.contains(doc.id) {
                    result[doc.id] = doc
                }
            }
        }

        return result
    }

    public func getDocumentCount() async throws -> Int {
        return try await getTotalDocumentCount()
    }

    public func storageSearch(vector: [Float], topK: Int) async throws -> [UUID]? {
        guard let indexedStorage = indexedStorage else {
            return nil
        }

        return try await indexedStorage.searchCandidates(
            queryEmbedding: vector,
            topK: topK,
            prefilterSize: topK * 10
        )
    }
}
