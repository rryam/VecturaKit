import Foundation

/// BM25 text search engine
///
/// This engine implements the BM25 ranking function for text search. It maintains
/// an in-memory index of documents and can optionally delegate to storage-layer
/// text search implementations (like SQLite FTS) if available.
///
/// ## Index Management
///
/// The index uses lazy loading: it's rebuilt from storage when needed (first search
/// or after being marked dirty). This provides a balance between performance and
/// memory efficiency.
///
/// **Note**: BM25 search always operates in full-memory mode. The index is built
/// from all documents in storage and kept in memory. This differs from vector search
/// which supports both in-memory and indexed strategies.
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
      index = BM25Index(documents: documents, k1: k1, b: b)
      needsRebuild = false
    }

    guard let index = index else {
      return []
    }

    let results = index.search(query: queryText, topK: options.numResults)

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
      // Index exists: update incrementally (zero-copy, direct modification)
      index.addDocument(document)
    } else {
      // Index not yet built: mark as needing rebuild on next search
      needsRebuild = true
    }
  }

  public func removeDocument(id: UUID) async throws {
    if let index = index {
      // Index exists: remove incrementally (zero-copy, direct modification)
      index.removeDocument(id)
    } else {
      // Index not yet built: mark as needing rebuild on next search
      needsRebuild = true
    }
  }
}
