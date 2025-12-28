import Foundation

/// BM25 text search engine
///
/// This engine implements the BM25 ranking function for text search. It maintains
/// an in-memory index of lightweight BM25Document objects.
///
/// ## Index Management
///
/// The index uses lazy loading: it's rebuilt from storage when needed (first search
/// or after being marked dirty). This provides a balance between performance and
/// memory efficiency.
///
/// ## Memory Efficiency
///
/// BM25SearchEngine uses lightweight BM25Document objects that store only:
/// - Document ID
/// - Text content
/// - Creation timestamp
///
/// This is significantly more memory-efficient than storing full VecturaDocument
/// objects with embeddings (~1.5KB savings per document for 384-dimensional vectors).
///
/// ## Index Unloading
///
/// After search operations, the index can be unloaded to free memory when using
/// indexed memory strategy. Call `unloadIndex()` to release memory.
///
/// ## Performance Characteristics
///
/// - **Index Building**: O(N × M) where N is document count, M is average token count
/// - **Search**: O(N × K) where K is query token count
/// - **Memory**: ~100-200 bytes per document plus inverted index overhead
///
/// For large datasets (>100K documents), consider implementing a storage provider
/// with native text search capabilities (e.g., SQLite FTS).
public actor BM25SearchEngine: VecturaSearchEngine {

  private var index: BM25Index?
  private var needsRebuild = false
  private let k1: Float
  private let b: Float

  /// Initialize BM25 search engine
  ///
  /// - Parameters:
  ///   - k1: BM25 k1 parameter (default: 1.2)
  ///   - b: BM25 b parameter (default: 0.75)
  public init(
    k1: Float = 1.2,
    b: Float = 0.75
  ) {
    self.k1 = k1
    self.b = b
  }

  // MARK: - VecturaSearchEngine Protocol

  public func search(
    query: SearchQuery,
    storage: VecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult] {
    guard case .text(let queryText) = query else {
      throw VecturaError.invalidInput("BM25 only supports text queries")
    }

    // Future enhancement: Detect storage-layer text search capability
    // Example:
    // if let textSearchable = storage as? TextSearchableStorage {
    //     return try await textSearchable.searchText(query: queryText, topK: options.numResults)
    // }

    // Rebuild index if needed (first search or after documents changed)
    if index == nil || needsRebuild {
      let documents = try await storage.loadDocuments()
      // Convert to lightweight BM25Document for memory efficiency
      let lightweightDocs = documents.map { BM25Document(from: $0) }
      index = BM25Index(documents: lightweightDocs, k1: k1, b: b)
      needsRebuild = false
    }

    guard let index = index else {
      return []
    }

    let results = await index.search(query: queryText, topK: options.numResults)

    // Filter by threshold
    var filteredResults = results
    if let threshold = options.threshold {
      filteredResults = results.filter { $0.score >= threshold }
    }

    return filteredResults.map { result in
      VecturaSearchResult(
        id: result.document.id,
        text: result.document.text,
        score: result.score,
        createdAt: result.document.createdAt
      )
    }
  }

  public func indexDocument(_ document: VecturaDocument) async throws {
    if let index = index {
      // Index exists: update incrementally with lightweight document
      await index.addDocument(BM25Document(from: document))
    } else {
      // Index not yet built: mark as needing rebuild on next search
      needsRebuild = true
    }
  }

  public func removeDocument(id: UUID) async throws {
    if let index = index {
      // Index exists: remove incrementally
      await index.removeDocument(id)
    } else {
      // Index not yet built: mark as needing rebuild on next search
      needsRebuild = true
    }
  }

  // MARK: - Index Management

  /// Unloads the BM25 index to free memory.
  ///
  /// After calling this method, the index will be cleared and will need to be
  /// rebuilt on the next search operation. This is useful when using indexed
  /// memory strategy and wanting to minimize memory footprint.
  ///
  /// Example:
  /// ```swift
  /// await bm25Engine.unloadIndex()
  /// ```
  public func unloadIndex() async {
    await index?.unload()
    index = nil
    needsRebuild = true
  }

  /// Returns whether the index is currently loaded in memory
  /// - Returns: True if index is loaded, false otherwise
  public var isIndexLoaded: Bool {
    index != nil
  }

  /// Returns the current number of documents in the index
  /// - Returns: The count of indexed documents, or 0 if not loaded
  public var indexedDocumentCount: Int {
    get async {
      await index?.documentCount ?? 0
    }
  }
}
