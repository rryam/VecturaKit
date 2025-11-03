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
    let normalizedQuery = try normalizeEmbedding(queryVector)

    // Build matrix of normalized document embeddings
    var docIds = [UUID]()
    var matrix = [Float]()
    matrix.reserveCapacity(documents.count * dimension)

    for doc in documents {
      let normalized = try normalizeEmbedding(doc.embedding)
      docIds.append(doc.id)
      matrix.append(contentsOf: normalized)
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
    return Array(results.prefix(options.numResults))
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
        options: SearchOptions(numResults: prefilterSize)
      )
      candidateIds = prefilterResults.map { $0.id }
    }

    guard !candidateIds.isEmpty else {
      return []
    }

    // Load candidate documents
    let candidates = try await indexedStorage.loadDocuments(ids: candidateIds)

    guard !candidates.isEmpty else {
      return []
    }

    // Compute exact similarities
    let normalizedQuery = try normalizeEmbedding(queryVector)
    let dimension = queryVector.count

    var candidateDocIds = [UUID]()
    var candidateDocs = [VecturaDocument]()
    var matrix = [Float]()
    matrix.reserveCapacity(candidates.count * dimension)

    for (id, doc) in candidates {
      let normalized = try normalizeEmbedding(doc.embedding)
      candidateDocIds.append(id)
      candidateDocs.append(doc)
      matrix.append(contentsOf: normalized)
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
    return Array(results.prefix(options.numResults))
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

  private func l2Norm(_ v: [Float]) -> Float {
    var sumSquares: Float = 0
    vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
    return sqrt(sumSquares)
  }
}
