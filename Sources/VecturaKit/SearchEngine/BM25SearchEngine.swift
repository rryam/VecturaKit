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

  /// Bumped on every mutation (add / remove / unload).
  ///
  /// A rebuild suspends at `storage.loadDocuments()`, so mutations can land
  /// while its snapshot is already fixed. Comparing this counter across the
  /// await identifies a snapshot that no longer reflects current state.
  private var mutationGeneration: UInt64 = 0

  /// The `mutationGeneration` the currently installed index reflects.
  ///
  /// Two overlapping rebuilds can resume in either order; without this, the
  /// slower one would clobber a newer index with its older snapshot and the
  /// documents added in between would be lost until the next unload.
  private var indexGeneration: UInt64 = 0

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

    if let indexedStorage = storage as? IndexedVecturaStorage {
      let indexedResults = try await indexedStorage.searchText(
        query: queryText,
        topK: options.numResults
      )
      if let indexedResults {
        return applyThreshold(options.threshold, to: indexedResults)
      }
    }

    // Rebuild the index if needed (first search, or documents changed).
    //
    // Clear the dirty flag before awaiting storage so that a mutation
    // interleaving during the load can re-mark it, and record the generation
    // this snapshot corresponds to so a slower concurrent rebuild cannot
    // install an older snapshot over a newer index.
    if index == nil || needsRebuild {
      needsRebuild = false
      let generationAtLoadStart = mutationGeneration

      let documents: [VecturaDocument]
      do {
        documents = try await storage.loadDocuments()
      } catch {
        needsRebuild = true
        throw error
      }

      // Install only if this snapshot is at least as fresh as what is already
      // installed. BM25Index handles conversion to lightweight BM25Document.
      if index == nil || indexGeneration <= generationAtLoadStart {
        index = BM25Index(documents: documents, k1: k1, b: b)
        indexGeneration = generationAtLoadStart

        // A mutation landed mid-load, so this snapshot may be missing it.
        // It still answers the current query; mark dirty to rebuild next time.
        if mutationGeneration != generationAtLoadStart {
          needsRebuild = true
        }
      }
    }

    guard let index = index else {
      return []
    }

    let results = await index.search(query: queryText, topK: options.numResults)

    let searchResults = results.map { result in
      VecturaSearchResult(
        id: result.document.id,
        text: result.document.text,
        score: result.score,
        createdAt: result.document.createdAt
      )
    }
    return applyThreshold(options.threshold, to: searchResults)
  }

  public func indexDocument(_ document: VecturaDocument) async throws {
    mutationGeneration &+= 1

    if let index = index {
      // Index handles conversion to lightweight BM25Document
      await index.addDocument(document)
    } else {
      // Index not yet built: mark as needing rebuild on next search
      needsRebuild = true
    }
  }

  public func removeDocument(id: UUID) async throws {
    mutationGeneration &+= 1

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
    // Detach and mark dirty *before* awaiting the unload. Awaiting first would
    // leave a non-nil but emptied index visible to a search that interleaves
    // here, which would skip the rebuild and return no results.
    let unloadedIndex = index
    index = nil
    needsRebuild = true
    mutationGeneration &+= 1

    await unloadedIndex?.unload()
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

  private func applyThreshold(
    _ threshold: Float?,
    to results: [VecturaSearchResult]
  ) -> [VecturaSearchResult] {
    guard let threshold else {
      return results
    }
    return results.filter { $0.score >= threshold }
  }
}
