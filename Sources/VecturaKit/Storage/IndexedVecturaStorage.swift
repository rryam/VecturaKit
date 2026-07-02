import Foundation

/// Extended storage protocol that supports indexed operations for large-scale datasets.
///
/// This protocol extends `VecturaStorage` with additional methods that enable efficient
/// handling of large document collections through pagination and indexed searching.
///
/// ## Performance Characteristics
///
/// Implementing this protocol allows VecturaKit to use an "indexed" memory strategy that:
/// - Loads documents on-demand rather than keeping everything in memory
/// - Supports efficient candidate filtering for vector search
/// - Scales to millions of documents with controlled memory usage
///
/// ## Implementation Notes
///
/// Storage providers that implement this protocol can enable VecturaKit's indexed mode,
/// which performs a two-stage search:
/// 1. **Candidate Selection**: Use `searchCandidates()` to identify promising document IDs
/// 2. **Exact Scoring**: Load only those candidates via `loadDocuments(ids:)` for precise ranking
///
/// This approach dramatically reduces memory footprint for large datasets while maintaining
/// search quality.
public protocol IndexedVecturaStorage: VecturaStorage {

  /// Loads a specific range of documents from storage.
  ///
  /// - Parameters:
  ///   - offset: The starting index (0-based)
  ///   - limit: Maximum number of documents to load
  /// - Returns: Array of documents in the specified range
  /// - Throws: Storage errors if loading fails
  func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument]

  /// Searches for candidate document IDs using an approximate vector search.
  ///
  /// This is an optional capability that storage providers can implement to enable
  /// efficient indexed vector search. If the storage doesn't support vector indexing,
  /// it should return `nil`, and the search engine will fall back to in-memory search.
  ///
  /// This is the first stage of indexed search. Implementations should use an
  /// approximate nearest neighbor (ANN) algorithm or similar technique to quickly
  /// identify promising candidates without loading full document objects.
  ///
  /// - Parameters:
  ///   - queryEmbedding: The query vector
  ///   - topK: Number of final results desired
  ///   - prefilterSize: Size of candidate pool (typically `topK * candidateMultiplier`)
  /// - Returns: Array of document IDs that are potential matches, or `nil` if not supported
  /// - Throws: Storage errors if search fails
  ///
  /// ## Implementation Strategy
  ///
  /// The `prefilterSize` parameter allows controlling the trade-off between accuracy
  /// and performance:
  /// - Larger values: Better recall but slower
  /// - Smaller values: Faster but may miss relevant results
  ///
  /// A typical implementation might:
  /// 1. Use a vector index (HNSW, IVF, etc.) to find `prefilterSize` candidates
  /// 2. Return their IDs for exact rescoring by the search engine
  ///
  /// ## Example
  ///
  /// ```swift
  /// // Storage with vector index support
  /// func searchVectorCandidates(...) async throws -> [UUID]? {
  ///     return try await queryVectorIndex(queryEmbedding, prefilterSize)
  /// }
  ///
  /// // Storage without vector index support
  /// func searchVectorCandidates(...) async throws -> [UUID]? {
  ///     return nil  // Fall back to in-memory search
  /// }
  /// ```
  func searchVectorCandidates(
    queryEmbedding: [Float],
    topK: Int,
    prefilterSize: Int
  ) async throws -> [UUID]?

  /// Searches text candidates using a storage-layer text index.
  ///
  /// This optional capability lets indexed storage providers answer BM25/text
  /// queries without forcing `BM25SearchEngine` to load every full document.
  /// Providers without native text indexing can return `nil`, allowing callers
  /// to fall back to their existing in-memory behavior.
  ///
  /// - Parameters:
  ///   - query: The text query to search.
  ///   - topK: Maximum number of ranked text results to return.
  /// - Returns: Ranked text results, or `nil` if storage-layer text search is unsupported.
  /// - Throws: Storage errors if text search fails.
  func searchText(
    query: String,
    topK: Int
  ) async throws -> [VecturaSearchResult]?

  /// Loads specific documents by their IDs.
  ///
  /// This is the second stage of indexed search, where only the candidate documents
  /// identified by `searchCandidates()` are loaded for exact similarity computation.
  ///
  /// - Parameter ids: Array of document IDs to load
  /// - Returns: Dictionary mapping IDs to their documents (may not include all requested IDs if some don't exist)
  /// - Throws: Storage errors if loading fails
  func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument]
}

extension IndexedVecturaStorage {
  public func searchText(
    query: String,
    topK: Int
  ) async throws -> [VecturaSearchResult]? {
    nil
  }
}
