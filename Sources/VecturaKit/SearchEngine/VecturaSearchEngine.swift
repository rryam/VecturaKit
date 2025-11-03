import Foundation

/// Protocol for search engine implementations
///
/// Search engines are responsible for implementing search algorithms (vector similarity,
/// text search, hybrid, etc.) and can optionally detect and use storage-layer capabilities
/// for optimization.
///
/// ## Storage Capability Detection
///
/// Search engines receive a `VecturaStorage` reference and can detect its capabilities
/// using type casting:
///
/// ```swift
/// func search(query: SearchQuery, storage: VecturaStorage, options: SearchOptions) async throws -> [VecturaSearchResult] {
///     // Detect indexed storage capability
///     if let indexedStorage = storage as? IndexedVecturaStorage {
///         // Use indexed operations
///         let candidateIds = try await indexedStorage.searchVectorCandidates(...)
///     }
///
///     // Fallback to basic storage operations
///     let documents = try await storage.loadDocuments()
/// }
/// ```
public protocol VecturaSearchEngine: Sendable {

  /// Execute search with the given query
  ///
  /// - Parameters:
  ///   - query: The search query (text or vector)
  ///   - storage: The storage provider (search engine can detect capabilities using 'as')
  ///   - options: Search configuration options
  /// - Returns: Array of search results sorted by relevance
  func search(
    query: SearchQuery,
    storage: VecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult]

  /// Index a document (optional, some engines don't need this)
  ///
  /// This method is called when documents are added or updated. Search engines
  /// that maintain their own indexes (like BM25) should update them here.
  ///
  /// - Parameter document: The document to index
  func indexDocument(_ document: VecturaDocument) async throws

  /// Remove a document from index (optional)
  ///
  /// This method is called when documents are deleted. Search engines that
  /// maintain their own indexes should clean them up here.
  ///
  /// - Parameter id: The ID of the document to remove
  func removeDocument(id: UUID) async throws
}
