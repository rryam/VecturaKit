import Accelerate
import Foundation

/// Vector search engine using VecturaKit's Accelerate-based similarity computation
public struct VectorSearchEngine: VecturaSearchEngine {

  /// The embedder for generating query vectors
  public let embedder: VecturaEmbedder

  /// Search strategy configuration
  public let strategy: VecturaConfig.MemoryStrategy

  public init(
    embedder: VecturaEmbedder,
    strategy: VecturaConfig.MemoryStrategy = .automatic()
  ) {
    self.embedder = embedder
    self.strategy = strategy
  }

  // MARK: - VecturaSearchEngine Protocol

  public func search(
    query: SearchQuery,
    storage: VecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult] {
    // Extract query vector
    let queryVector: [Float]
    switch query {
    case .vector(let vec):
      queryVector = vec
    case .text(let text):
      queryVector = try await embedder.embed(text: text)
    }

    // Choose search strategy based on configuration and storage capabilities
    let shouldUseIndexed = try await shouldUseIndexedSearch(storage: storage)

    if shouldUseIndexed, let indexedStorage = storage as? IndexedVecturaStorage {
      return try await searchWithIndexedStorage(
        queryVector: queryVector,
        indexedStorage: indexedStorage,
        options: options
      )
    }

    // Fallback: in-memory search
    return try await searchInMemory(
      queryVector: queryVector,
      storage: storage,
      options: options
    )
  }

  public func indexDocument(_ document: VecturaDocument) async throws {
    // Vector search doesn't need additional indexing
  }

  public func removeDocument(id: UUID) async throws {
    // Vector search doesn't need additional cleanup
  }

  // MARK: - Search Implementations

  private func searchInMemory(
    queryVector: [Float],
    storage: VecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult] {
    let documents = try await storage.loadDocuments()

    guard !documents.isEmpty else {
      return []
    }

    let dimension = queryVector.count

    // Normalize query vector
    let normalizedQuery = try VectorMath.normalizeEmbedding(queryVector)

    // Build matrix of document embeddings (already normalized at storage time)
    var docIds = [UUID]()
    var matrix = [Float]()
    matrix.reserveCapacity(documents.count * dimension)

    for doc in documents {
      docIds.append(doc.id)
      matrix.append(contentsOf: doc.embedding)
    }

    let docsCount = docIds.count
    var similarities = [Float](repeating: 0, count: docsCount)

    // Compute similarities using matrix-vector multiplication
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

    // Build results
    var results = [VecturaSearchResult]()
    results.reserveCapacity(docsCount)

    for (i, similarity) in similarities.enumerated() {
      if let threshold = options.threshold, similarity < threshold {
        continue
      }

      // docIds and documents are built in parallel, so indices correspond
      let doc = documents[i]
      results.append(
        VecturaSearchResult(
          id: doc.id,
          text: doc.text,
          score: similarity,
          createdAt: doc.createdAt
        )
      )
    }

    results.sort { $0.score > $1.score }
    if results.count > options.numResults {
      results.removeSubrange(options.numResults..<results.count)
    }
    return results
  }

  private func searchWithIndexedStorage(
    queryVector: [Float],
    indexedStorage: IndexedVecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult] {
    let candidateMultiplier = getCandidateMultiplier()
    let prefilterSize = options.numResults * candidateMultiplier

    // Try storage-layer vector search first
    let candidateIds: [UUID]
    if let storageIds = try await indexedStorage.searchVectorCandidates(
      queryEmbedding: queryVector,
      topK: options.numResults,
      prefilterSize: prefilterSize
    ) {
      candidateIds = storageIds
    } else {
      // Fallback: load all documents and do in-memory prefilter
      _ = try await indexedStorage.loadDocuments()  // Ensure documents loaded
      let prefilterResults = try await searchInMemory(
        queryVector: queryVector,
        storage: indexedStorage,
        options: try SearchOptions(numResults: prefilterSize)
      )
      candidateIds = prefilterResults.map { $0.id }
    }

    guard !candidateIds.isEmpty else {
      return []
    }

    // Load candidate documents with batching for memory efficiency
    let candidates = try await loadCandidatesWithBatching(
      candidateIds: candidateIds,
      storage: indexedStorage
    )

    guard !candidates.isEmpty else {
      return []
    }

    // Compute exact similarities
    let normalizedQuery = try VectorMath.normalizeEmbedding(queryVector)
    let dimension = queryVector.count

    var candidateDocIds = [UUID]()
    var candidateDocs = [VecturaDocument]()
    var matrix = [Float]()
    matrix.reserveCapacity(candidates.count * dimension)

    // Document embeddings are already normalized at storage time
    for (id, doc) in candidates {
      candidateDocIds.append(id)
      candidateDocs.append(doc)
      matrix.append(contentsOf: doc.embedding)
    }

    let candidatesCount = candidateDocIds.count
    var similarities = [Float](repeating: 0, count: candidatesCount)

    cblas_sgemv(
      CblasRowMajor,
      CblasNoTrans,
      candidatesCount,
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

    // Build results
    var results = [VecturaSearchResult]()
    results.reserveCapacity(candidatesCount)

    for (i, similarity) in similarities.enumerated() {
      if let threshold = options.threshold, similarity < threshold {
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

    results.sort { $0.score > $1.score }
    if results.count > options.numResults {
      results.removeSubrange(options.numResults..<results.count)
    }
    return results
  }

  // MARK: - Helper Methods

  private func shouldUseIndexedSearch(storage: VecturaStorage) async throws -> Bool {
    switch strategy {
    case .fullMemory:
      return false
    case .indexed:
      return true
    case .automatic(let threshold, _, _, _):
      // Get document count efficiently using storage's getTotalDocumentCount
      let count = try await storage.getTotalDocumentCount()
      return count >= threshold
    }
  }

  private func getCandidateMultiplier() -> Int {
    switch strategy {
    case .indexed(let multiplier, _, _):
      return multiplier
    case .automatic(_, let multiplier, _, _):
      return multiplier
    case .fullMemory:
      return 10  // Default multiplier if somehow used in fullMemory mode
    }
  }


  // MARK: - Batch Loading

  /// Loads candidate documents using batched loading for memory efficiency.
  ///
  /// This method implements controlled batching to prevent memory spikes when loading
  /// large numbers of candidate documents in indexed mode.
  ///
  /// - Parameters:
  ///   - candidateIds: Array of document IDs to load
  ///   - storage: The indexed storage provider
  /// - Returns: Dictionary mapping document IDs to their documents
  /// - Throws: VecturaError if all batches fail
  private func loadCandidatesWithBatching(
    candidateIds: [UUID],
    storage: IndexedVecturaStorage
  ) async throws -> [UUID: VecturaDocument] {
    // Extract batch parameters from strategy
    let batchSize: Int
    let maxConcurrentBatches: Int
    switch strategy {
    case .indexed(_, let extractedBatchSize, let extractedMaxConcurrent):
      batchSize = extractedBatchSize
      maxConcurrentBatches = extractedMaxConcurrent
    case .automatic(_, _, let extractedBatchSize, let extractedMaxConcurrent):
      batchSize = extractedBatchSize
      maxConcurrentBatches = extractedMaxConcurrent
    case .fullMemory:
      batchSize = VecturaConfig.MemoryStrategy.defaultBatchSize
      maxConcurrentBatches = VecturaConfig.MemoryStrategy.defaultMaxConcurrentBatches
    }

    // For small candidate sets, load directly without batching overhead
    guard candidateIds.count > batchSize else {
      return try await storage.loadDocuments(ids: candidateIds)
    }

    // Split IDs into batches
    let batches = stride(from: 0, to: candidateIds.count, by: batchSize).map { startIndex in
      let endIndex = min(startIndex + batchSize, candidateIds.count)
      return Array(candidateIds[startIndex..<endIndex])
    }

    // Load batches with controlled concurrency
    var allDocuments: [UUID: VecturaDocument] = [:]
    var failureCount = 0

    // Process batches in groups to limit concurrency
    for batchGroup in batches.chunked(into: maxConcurrentBatches) {
      await withTaskGroup(of: (success: Bool, docs: [UUID: VecturaDocument]).self) { group in
        for batch in batchGroup {
          group.addTask {
            do {
              let docs = try await storage.loadDocuments(ids: batch)
              return (success: true, docs: docs)
            } catch {
              return (success: false, docs: [:])
            }
          }
        }

        for await result in group {
          if result.success {
            allDocuments.merge(result.docs) { _, new in new }
          } else {
            failureCount += 1
          }
        }
      }
    }

    // Only throw if we got NO results at all
    if allDocuments.isEmpty && failureCount > 0 {
      throw VecturaError.loadFailed(
        "Failed to load any candidate documents (\(failureCount) batch(es) failed)"
      )
    }

    return allDocuments
  }
}
