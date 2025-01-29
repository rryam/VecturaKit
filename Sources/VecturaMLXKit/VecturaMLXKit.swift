import Foundation
import VecturaKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class VecturaMLXKit {
  private let config: VecturaConfig
  private let embedder: MLXEmbedder
  private var documents: [UUID: VecturaDocument] = [:]
  private var normalizedEmbeddings: [UUID: [Float]] = [:]

  public init(config: VecturaConfig, modelConfiguration: ModelConfiguration = .nomic_text_v1_5)
    async throws
  {
    self.config = config
    self.embedder = try await MLXEmbedder(configuration: modelConfiguration)
  }

  public func addDocuments(texts: [String], ids: [UUID]? = nil) async throws -> [UUID] {
    if let ids = ids, ids.count != texts.count {
      throw VecturaError.invalidInput("Number of IDs must match number of texts")
    }

    let embeddings = await embedder.embed(texts: texts)
    var documentIds = [UUID]()

    for (index, text) in texts.enumerated() {
      let docId = ids?[index] ?? UUID()
      let doc = VecturaDocument(id: docId, text: text, embedding: embeddings[index])

      // Normalize embedding for cosine similarity
      let norm = l2Norm(doc.embedding)
      let normalized = doc.embedding.map { $0 / (norm + 1e-9) }

      normalizedEmbeddings[doc.id] = normalized
      documents[doc.id] = doc
      documentIds.append(docId)
    }

    return documentIds
  }

  public func search(query: String, numResults: Int? = nil, threshold: Float? = nil) async throws
    -> [VecturaSearchResult]
  {
    let queryEmbedding = try await embedder.embed(text: query)

    let norm = l2Norm(queryEmbedding)
    let normalizedQuery = queryEmbedding.map { $0 / (norm + 1e-9) }

    var results: [VecturaSearchResult] = []

    for doc in documents.values {
      guard let normDoc = normalizedEmbeddings[doc.id] else { continue }
      let similarity = dotProduct(normalizedQuery, normDoc)

      if let minT = threshold ?? config.searchOptions.minThreshold, similarity < minT {
        continue
      }

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

    let limit = numResults ?? config.searchOptions.defaultNumResults
    return Array(results.prefix(limit))
  }

  public func reset() {
    documents.removeAll()
    normalizedEmbeddings.removeAll()
  }

  // MARK: - Private

  private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).reduce(into: 0) { $0 += $1.0 * $1.1 }
  }

  private func l2Norm(_ v: [Float]) -> Float {
    sqrt(dotProduct(v, v))
  }
}
