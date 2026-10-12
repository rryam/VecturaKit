import Foundation
import Testing
@testable import VecturaKit

@Suite("Query embedding dispatch")
struct QueryEmbeddingTests {
  private actor QueryProbe: VecturaEmbedder {
    private var documents: [String] = []
    private var queries: [String] = []
    var dimension: Int { get async throws { 2 } }

    func embed(texts: [String]) async throws -> [[Float]] {
      documents.append(contentsOf: texts)
      return texts.map { $0 == "right" ? [0, 1] : [1, 0] }
    }

    func embedQuery(text: String) async throws -> [Float] {
      queries.append(text)
      return [0, 1]
    }

    func snapshot() -> (documents: [String], queries: [String]) { (documents, queries) }
  }

  private actor LegacySingleTextProbe: VecturaEmbedder {
    private var batchCalls = 0
    private var singleCalls = 0
    var dimension: Int { get async throws { 2 } }

    func embed(texts: [String]) async throws -> [[Float]] {
      batchCalls += 1
      return Array(repeating: [0, 1], count: texts.count)
    }

    func embed(text: String) async throws -> [Float] {
      singleCalls += 1
      return [1, 0]
    }

    func counts() -> (batch: Int, single: Int) { (batchCalls, singleCalls) }
  }

  private struct FailingQueryEmbedder: VecturaEmbedder {
    var dimension: Int { get async throws { 2 } }
    func embed(texts: [String]) async throws -> [[Float]] { Array(repeating: [1, 0], count: texts.count) }
    func embedQuery(text: String) async throws -> [Float] { throw CancellationError() }
  }

  @Test("Add/update use document embedding; text searches use the query override",
    arguments: [false, true])
  func databaseDispatch(_ hybrid: Bool) async throws {
    let directory = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let embedder = QueryProbe()
    let config = try VecturaConfig(name: "dispatch", directoryURL: directory,
      searchOptions: .init(minThreshold: 0.5, hybridWeight: 1), memoryStrategy: .fullMemory)
    let engine: (any VecturaSearchEngine)? = hybrid ? nil : VectorSearchEngine(embedder: embedder)
    let database = try await VecturaKit(config: config, embedder: embedder, searchEngine: engine)
    let identifiers = try await database.addDocuments(texts: ["left", "right"])
    try await database.updateDocument(id: identifiers[0], newText: "updated")
    let results = try await database.search(query: .text("left"), numResults: 2)
    #expect(results.count == 1)
    #expect(results.first?.id == identifiers[1]) // Query vector differs from the document embedding of "left".
    #expect(results.first?.score == 1)
    let vectorResults = try await database.search(query: .vector([1, 0]), numResults: 2)
    #expect(vectorResults.first?.id == identifiers[0])
    let calls = await embedder.snapshot()
    #expect(calls.documents == ["left", "right", "updated"])
    #expect(calls.queries == ["left"]) // Vector queries bypass both embedding methods.
    let updated = try #require(try await database.getDocument(id: identifiers[0]))
    #expect(updated.embedding == [1, 0])
  }

  @Test("Default query embedding dispatches to an existing single-text override")
  func legacySingleTextFallback() async throws {
    let embedder = LegacySingleTextProbe()
    let query = try await embedder.embedQuery(text: "query")
    #expect(query == [1, 0])
    let counts = await embedder.counts()
    #expect(counts.batch == 0)
    #expect(counts.single == 1)
    #expect(embedder.embeddingSpaceIdentifier == nil)
  }

  @Test("Both search engines propagate query-hook cancellation", arguments: [false, true])
  func queryCancellation(_ hybrid: Bool) async throws {
    let embedder = FailingQueryEmbedder()
    let storage = InMemoryStorageProvider()
    let vectorEngine = VectorSearchEngine(embedder: embedder)
    let engine: any VecturaSearchEngine = hybrid
      ? HybridSearchEngine(vectorEngine: vectorEngine, textEngine: BM25SearchEngine()) : vectorEngine
    await #expect(throws: CancellationError.self) {
      try await engine.search(query: .text("query"), storage: storage, options: SearchOptions())
    }
  }
}
