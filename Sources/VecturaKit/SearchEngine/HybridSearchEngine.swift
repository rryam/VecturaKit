import Foundation

/// Hybrid search engine that combines vector similarity and text search
///
/// This engine performs hybrid search by:
/// 1. Running vector search and text search concurrently
/// 2. Combining their scores with configurable weights
/// 3. Returning unified results sorted by the combined score
///
/// ## Query Type Behavior
///
/// - `.text(query)`: Performs hybrid search by embedding the text and running both
///   vector and text searches in parallel
/// - `.vector(embedding)`: Falls back to pure vector search since there's no text
///   to perform text search on
///
/// ## Performance Characteristics
///
/// - Retrieves 2x the requested results from each engine to ensure high-quality
///   hybrid ranking
/// - Uses concurrent execution (`async let`) for minimal latency
///
/// ## Example
///
/// ```swift
/// let hybrid = HybridSearchEngine(
///   vectorEngine: vectorEngine,
///   textEngine: bm25Engine,
///   vectorWeight: 0.7  // 70% vector, 30% text
/// )
///
/// // Hybrid search (uses both engines)
/// let results = try await hybrid.search(
///   query: .text("machine learning"),
///   storage: storage,
///   options: options
/// )
///
/// // Pure vector search (only uses vector engine)
/// let vectorResults = try await hybrid.search(
///   query: .vector(embedding),
///   storage: storage,
///   options: options
/// )
/// ```
public struct HybridSearchEngine: VecturaSearchEngine {

  private let vectorEngine: VectorSearchEngine
  private let textEngine: any VecturaSearchEngine
  private let vectorWeight: Float

  /// Initialize hybrid search engine
  /// - Parameters:
  ///   - vectorEngine: Vector search engine for semantic similarity
  ///   - textEngine: Text search engine (e.g., BM25 or SQLite FTS)
  ///   - vectorWeight: Weight for vector score (0.0-1.0), text weight will be (1 - vectorWeight)
  public init(
    vectorEngine: VectorSearchEngine,
    textEngine: any VecturaSearchEngine,
    vectorWeight: Float = 0.5
  ) {
    self.vectorEngine = vectorEngine
    self.textEngine = textEngine
    self.vectorWeight = max(0, min(1, vectorWeight))
  }

  // MARK: - VecturaSearchEngine Protocol

  public func search(
    query: SearchQuery,
    storage: VecturaStorage,
    options: SearchOptions
  ) async throws -> [VecturaSearchResult] {
    let queryText: String
    let queryVector: [Float]

    switch query {
    case .text(let text):
      queryText = text
      queryVector = try await vectorEngine.embedder.embed(text: text)
    case .vector:
      // Pure vector query, fallback to vector search
      return try await vectorEngine.search(
        query: query,
        storage: storage,
        options: options
      )
    }

    // Execute vector and text search concurrently
    let extendedOptions = SearchOptions(
      numResults: options.numResults * 2,  // Get more candidates
      threshold: nil  // Don't apply threshold in individual searches
    )

    async let vectorResults = vectorEngine.search(
      query: .vector(queryVector),
      storage: storage,
      options: extendedOptions
    )

    async let textResults = textEngine.search(
      query: .text(queryText),
      storage: storage,
      options: extendedOptions
    )

    // Combine results
    let vResults = try await vectorResults
    let tResults = try await textResults

    return combineResults(
      vectorResults: vResults,
      textResults: tResults,
      vectorWeight: vectorWeight,
      threshold: options.threshold,
      topK: options.numResults
    )
  }

  public func indexDocument(_ document: VecturaDocument) async throws {
    // Only need to notify text search engine
    try await textEngine.indexDocument(document)
  }

  public func removeDocument(id: UUID) async throws {
    // Only need to notify text search engine
    try await textEngine.removeDocument(id: id)
  }

  // MARK: - Private Methods

  private func combineResults(
    vectorResults: [VecturaSearchResult],
    textResults: [VecturaSearchResult],
    vectorWeight: Float,
    threshold: Float?,
    topK: Int
  ) -> [VecturaSearchResult] {
    // Build text search score map
    let textScores = Dictionary(
      textResults.map { ($0.id, $0.score) },
      uniquingKeysWith: { first, _ in first }
    )

    // Calculate hybrid scores
    var combinedResults: [VecturaSearchResult] = []
    var seenIds: Set<UUID> = []

    // Start from vector results
    for result in vectorResults {
      let textScore = textScores[result.id] ?? 0
      let hybridScore = vectorWeight * result.score + (1 - vectorWeight) * textScore

      // Apply threshold
      if let threshold = threshold, hybridScore < threshold {
        continue
      }

      combinedResults.append(VecturaSearchResult(
        id: result.id,
        text: result.text,
        score: hybridScore,
        createdAt: result.createdAt
      ))
      seenIds.insert(result.id)
    }

    // Add results only in text search
    for result in textResults where !seenIds.contains(result.id) {
      let hybridScore = (1 - vectorWeight) * result.score

      // Apply threshold
      if let threshold = threshold, hybridScore < threshold {
        continue
      }

      combinedResults.append(VecturaSearchResult(
        id: result.id,
        text: result.text,
        score: hybridScore,
        createdAt: result.createdAt
      ))
    }

    // Sort and return top K
    combinedResults.sort { $0.score > $1.score }
    return Array(combinedResults.prefix(topK))
  }
}
