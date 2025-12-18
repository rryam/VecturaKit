import Foundation
import Testing
@testable import VecturaKit

@Suite("HybridSearchEngine")
struct HybridSearchEngineTests {

  private struct FixedEmbedder: VecturaEmbedder {
    let embedding: [Float]

    var dimension: Int {
      get async throws { embedding.count }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      Array(repeating: embedding, count: texts.count)
    }
  }

  private actor FixedStorage: VecturaStorage {
    let documents: [VecturaDocument]

    init(documents: [VecturaDocument]) {
      self.documents = documents
    }

    func createStorageDirectoryIfNeeded() async throws {}
    func loadDocuments() async throws -> [VecturaDocument] { documents }
    func saveDocument(_ document: VecturaDocument) async throws {}
    func deleteDocument(withID id: UUID) async throws {}
    func updateDocument(_ document: VecturaDocument) async throws {}
  }

  private struct StubTextEngine: VecturaSearchEngine {
    let results: [VecturaSearchResult]

    func search(
      query: SearchQuery,
      storage: VecturaStorage,
      options: SearchOptions
    ) async throws -> [VecturaSearchResult] {
      results
    }

    func indexDocument(_ document: VecturaDocument) async throws {}
    func removeDocument(id: UUID) async throws {}
  }

  @Test("Hybrid search normalizes BM25 scores")
  func hybridNormalizesBm25Scores() async throws {
    let createdAt = Date()
    let docId = UUID()
    let doc = VecturaDocument(
      id: docId,
      text: "doc",
      embedding: [1.0, 0.0],
      createdAt: createdAt
    )

    let storage = FixedStorage(documents: [doc])
    let vectorEngine = VectorSearchEngine(
      embedder: FixedEmbedder(embedding: [1.0, 0.0]),
      strategy: .fullMemory
    )
    let textEngine = StubTextEngine(
      results: [
        VecturaSearchResult(
          id: docId,
          text: doc.text,
          score: 5.0,
          createdAt: createdAt
        )
      ]
    )
    let hybrid = HybridSearchEngine(
      vectorEngine: vectorEngine,
      textEngine: textEngine,
      vectorWeight: 0.5,
      bm25NormalizationFactor: 10.0
    )

    let results = try await hybrid.search(
      query: .text("query"),
      storage: storage,
      options: try SearchOptions(numResults: 10)
    )

    #expect(results.count == 1)
    #expect(abs(results[0].score - 0.75) < 0.0001)
  }
}
