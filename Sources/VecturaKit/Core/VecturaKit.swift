import Accelerate
import Foundation
import OSLog

/// A vector database implementation that stores and searches documents using their vector embeddings.
///
/// VecturaKit coordinates between storage providers and search engines, handling document
/// lifecycle management (add, update, delete) while delegating search operations to pluggable
/// search engines and persistence to pluggable storage providers.
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

  /// The dimension of vectors from the embedder (validated against config if specified).
  private let dimension: Int

  /// The storage provider that handles document persistence (and optionally caching).
  private let storageProvider: VecturaStorage

  // MARK: - Initialization

  /// Initializes a new `VecturaKit` instance with the specified configuration and embedder.
  ///
  /// - Parameters:
  ///   - config: Configuration options for the vector database.
  ///   - embedder: The embedder to use for generating vector embeddings from text.
  ///   - storageProvider: Optional custom storage provider. If nil, uses FileStorageProvider with caching enabled.
  ///   - searchEngine: Optional custom search engine. If nil, uses default hybrid search based on config.
  public init(
    config: VecturaConfig,
    embedder: VecturaEmbedder,
    storageProvider: VecturaStorage? = nil,
    searchEngine: VecturaSearchEngine? = nil
  ) async throws {
    self.config = config
    self.embedder = embedder

    // Determine dimension: prefer config if specified, otherwise use embedder
    if let configDimension = config.dimension {
      self.dimension = configDimension
    } else {
      self.dimension = try await embedder.dimension
    }

    // Use custom storage provider if provided, otherwise create FileStorageProvider
    if let customProvider = storageProvider {
      self.storageProvider = customProvider
    } else {
      // Create storage directory only when using default FileStorageProvider
      let storageDirectory = try Self.createStorageDirectory(config: config)
      self.storageProvider = try FileStorageProvider(storageDirectory: storageDirectory, cacheEnabled: true)
    }

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
    // Validate input
    guard !texts.isEmpty else {
      throw VecturaError.invalidInput("Cannot add empty array of documents")
    }

    // Validate that no text is empty
    for (index, text) in texts.enumerated() {
      guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw VecturaError.invalidInput("Document at index \(index) cannot be empty or whitespace-only")
      }
    }

    if let ids = ids, ids.count != texts.count {
      throw VecturaError.invalidInput("Number of IDs must match number of texts")
    }

    // Get embeddings from the embedder
    let embeddings = try await embedder.embed(texts: texts)

    // Validate embeddings
    for embedding in embeddings {
      try validateDimension(embedding)
    }

    var documentIds = [UUID]()
    var documentsToSave = [VecturaDocument]()

    for i in 0..<texts.count {
      let docId = ids?[i] ?? UUID()

      // Pre-normalize embedding at storage time to avoid per-search normalization
      let normalizedEmbedding = try normalizeEmbedding(embeddings[i])

      let doc = VecturaDocument(
        id: docId,
        text: texts[i],
        embedding: normalizedEmbedding
      )
      documentsToSave.append(doc)
      documentIds.append(docId)
    }

    // Save documents to storage (storage handles caching internally)
    let storage = self.storageProvider
    try await withThrowingTaskGroup(of: Void.self) { group in
      for doc in documentsToSave {
        group.addTask {
          try await storage.saveDocument(doc)
        }
      }

      try await group.waitForAll()
    }

    // Notify search engine to index documents
    for doc in documentsToSave {
      try await searchEngine.indexDocument(doc)
    }

    return documentIds
  }

  /// Searches for similar documents using the specified query.
  ///
  /// - Parameters:
  ///   - query: The search query (text or vector)
  ///   - numResults: Maximum number of results to return
  ///   - threshold: Minimum similarity threshold (0.0-1.0)
  /// - Returns: An array of search results ordered by relevance
  ///
  /// ## Examples
  ///
  /// Text search:
  /// ```swift
  /// let results = try await vectura.search(query: "machine learning")
  /// ```
  ///
  /// Vector search:
  /// ```swift
  /// let embedding: [Float] = [0.1, 0.2, ...]
  /// let results = try await vectura.search(query: .vector(embedding))
  /// ```
  public func search(
    query: SearchQuery,
    numResults: Int? = nil,
    threshold: Float? = nil
  ) async throws -> [VecturaSearchResult] {
    // Validate vector dimensions for vector queries
    switch query {
    case .vector(let embedding):
      try validateDimension(embedding)
    case .text:
      break  // Text queries don't need validation - embedder generates correct dimension
    }

    let options = try SearchOptions(
      numResults: numResults ?? config.searchOptions.defaultNumResults,
      threshold: threshold ?? config.searchOptions.minThreshold
    )

    return try await searchEngine.search(
      query: query,
      storage: storageProvider,
      options: options
    )
  }

  /// Removes all documents from the vector store.
  public func reset() async throws {
    // Get all document IDs from storage
    let allDocs = try await storageProvider.loadDocuments()
    let documentIDs = allDocs.map(\.id)
    try await deleteDocuments(ids: documentIDs)
  }

  /// Deletes specific documents from the vector store.
  ///
  /// - Parameter ids: The IDs of documents to delete.
  public func deleteDocuments(ids: [UUID]) async throws {
    for id in ids {
      // Delete using storage provider (storage handles cache updates)
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
    // Load document from storage
    let oldDocument: VecturaDocument
    if let indexed = storageProvider as? IndexedVecturaStorage {
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

    // Generate new embedding
    let newEmbedding = try await embedder.embed(text: newText)

    // Validate dimension
    try validateDimension(newEmbedding)

    // Pre-normalize embedding at storage time to avoid per-search normalization
    let normalizedEmbedding = try normalizeEmbedding(newEmbedding)

    // Create updated document, preserving original creation date
    let updatedDoc = VecturaDocument(
      id: id,
      text: newText,
      embedding: normalizedEmbedding,
      createdAt: oldDocument.createdAt
    )

    // Persist the updated document (storage handles cache updates)
    try await storageProvider.saveDocument(updatedDoc)

    // Notify search engine
    try await searchEngine.removeDocument(id: id)
    try await searchEngine.indexDocument(updatedDoc)
  }

  // MARK: - Public Properties

  /// Returns the number of documents currently stored in the vector database.
  public var documentCount: Int {
    get async throws {
      return try await storageProvider.getTotalDocumentCount()
    }
  }

  /// Returns all documents currently stored in the vector database.
  ///
  /// - Returns: Array of `VecturaDocument` objects
  public func getAllDocuments() async throws -> [VecturaDocument] {
    return try await storageProvider.loadDocuments()
  }

  // MARK: - Private

  /// Creates and returns the storage directory URL based on configuration
  private static func createStorageDirectory(config: VecturaConfig) throws -> URL {
    let storageDirectory: URL

    if let customStorageDirectory = config.directoryURL {
      storageDirectory = customStorageDirectory.appending(path: config.name)
    } else {
      // Create default storage directory
      guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
        throw VecturaError.loadFailed("Could not access document directory")
      }
      storageDirectory = documentsURL
        .appendingPathComponent("VecturaKit")
        .appendingPathComponent(config.name)
    }

    // Create directory if it doesn't exist
    if !FileManager.default.fileExists(atPath: storageDirectory.path(percentEncoded: false)) {
      try FileManager.default.createDirectory(
        at: storageDirectory,
        withIntermediateDirectories: true
      )
    }

    return storageDirectory
  }

  /// Creates default search engine based on configuration
  private static func createDefaultSearchEngine(
    config: VecturaConfig,
    embedder: VecturaEmbedder
  ) -> VecturaSearchEngine {
    // Create vector search engine with memory strategy
    let vectorEngine = VectorSearchEngine(
      embedder: embedder,
      strategy: config.memoryStrategy
    )

    // Create BM25 text search engine
    let bm25Engine = BM25SearchEngine(
      k1: config.searchOptions.k1,
      b: config.searchOptions.b
    )

    // Combine into hybrid search engine
    return HybridSearchEngine(
      vectorEngine: vectorEngine,
      textEngine: bm25Engine,
      vectorWeight: config.searchOptions.hybridWeight
    )
  }

  /// Validates that the embedding dimension matches the expected dimension
  private func validateDimension(_ embedding: [Float]) throws {
    if embedding.count != dimension {
      throw VecturaError.dimensionMismatch(
        expected: dimension,
        got: embedding.count
      )
    }
  }

  /// Normalizes an embedding vector to unit length (L2 normalization)
  private func normalizeEmbedding(_ embedding: [Float]) throws -> [Float] {
    let norm = l2Norm(embedding)

    guard norm > 1e-10 else {
      throw VecturaError.invalidInput("Cannot normalize zero-norm embedding vector")
    }

    var divisor = norm
    var normalized = [Float](repeating: 0, count: embedding.count)
    vDSP_vsdiv(embedding, 1, &divisor, &normalized, 1, vDSP_Length(embedding.count))
    return normalized
  }

  /// Computes the L2 norm of a vector
  private func l2Norm(_ v: [Float]) -> Float {
    var sumSquares: Float = 0
    vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
    return sqrt(sumSquares)
  }
}
