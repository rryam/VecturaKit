import Foundation
import Testing
@testable import VecturaKit

@Suite("Indexed Search Paths")
struct IndexedSearchPathTests {

  private struct IndexedStorageSnapshot: Sendable {
    let searchVectorCandidatesCalls: Int
    let lastTopK: Int?
    let lastPrefilterSize: Int?
    let loadDocumentsCalled: Bool
    let loadDocumentsIdsCalls: Int
    let lastLoadedIds: [UUID]
  }

  private struct FixedEmbedder: VecturaEmbedder {
    let embedding: [Float]

    var dimension: Int {
      get async throws { embedding.count }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      Array(repeating: embedding, count: texts.count)
    }
  }

  private actor IndexedStorageSpy: IndexedVecturaStorage {
    private let documents: [VecturaDocument]
    private let candidateIdsToReturn: [UUID]?
    private var searchVectorCandidatesCalls = 0
    private var lastTopK: Int?
    private var lastPrefilterSize: Int?
    private var loadDocumentsCalled = false
    private var loadDocumentsIdsCalls = 0
    private var lastLoadedIds: [UUID] = []

    init(documents: [VecturaDocument], candidateIdsToReturn: [UUID]?) {
      self.documents = documents
      self.candidateIdsToReturn = candidateIdsToReturn
    }

    func snapshot() -> IndexedStorageSnapshot {
      IndexedStorageSnapshot(
        searchVectorCandidatesCalls: searchVectorCandidatesCalls,
        lastTopK: lastTopK,
        lastPrefilterSize: lastPrefilterSize,
        loadDocumentsCalled: loadDocumentsCalled,
        loadDocumentsIdsCalls: loadDocumentsIdsCalls,
        lastLoadedIds: lastLoadedIds
      )
    }

    func createStorageDirectoryIfNeeded() async throws {}

    func loadDocuments() async throws -> [VecturaDocument] {
      loadDocumentsCalled = true
      return documents
    }

    func saveDocument(_ document: VecturaDocument) async throws {}

    func deleteDocument(withID id: UUID) async throws {}

    func updateDocument(_ document: VecturaDocument) async throws {}

    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument] {
      Array(documents.dropFirst(offset).prefix(limit))
    }

    func searchVectorCandidates(
      queryEmbedding: [Float],
      topK: Int,
      prefilterSize: Int
    ) async throws -> [UUID]? {
      searchVectorCandidatesCalls += 1
      lastTopK = topK
      lastPrefilterSize = prefilterSize
      return candidateIdsToReturn
    }

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
      loadDocumentsIdsCalls += 1
      lastLoadedIds = ids
      return Dictionary(
        uniqueKeysWithValues: documents.filter { ids.contains($0.id) }.map { ($0.id, $0) }
      )
    }
  }

  @Test("Uses storage candidate search when available")
  func usesStorageCandidateSearch() async throws {
    let doc1 = VecturaDocument(id: UUID(), text: "doc1", embedding: [1.0, 0.0])
    let doc2 = VecturaDocument(id: UUID(), text: "doc2", embedding: [1.0, 0.0])
    let storage = IndexedStorageSpy(documents: [doc1, doc2], candidateIdsToReturn: [doc2.id])
    let engine = VectorSearchEngine(
      embedder: FixedEmbedder(embedding: [1.0, 0.0]),
      strategy: .indexed(candidateMultiplier: 2, batchSize: 10, maxConcurrentBatches: 1)
    )

    _ = try await engine.search(
      query: .vector([1.0, 0.0]),
      storage: storage,
      options: try SearchOptions(numResults: 1)
    )

    let snapshot = await storage.snapshot()
    #expect(snapshot.searchVectorCandidatesCalls == 1)
    #expect(snapshot.lastTopK == 1)
    #expect(snapshot.lastPrefilterSize == 2)
    #expect(snapshot.loadDocumentsCalled == false)
    #expect(snapshot.loadDocumentsIdsCalls == 1)
    #expect(snapshot.lastLoadedIds == [doc2.id])
  }

  @Test("Falls back when candidate search is unsupported")
  func fallsBackWhenCandidateSearchUnsupported() async throws {
    let doc1 = VecturaDocument(id: UUID(), text: "doc1", embedding: [1.0, 0.0])
    let doc2 = VecturaDocument(id: UUID(), text: "doc2", embedding: [0.8, 0.6])
    let doc3 = VecturaDocument(id: UUID(), text: "doc3", embedding: [0.0, 1.0])
    let storage = IndexedStorageSpy(
      documents: [doc1, doc2, doc3],
      candidateIdsToReturn: nil
    )
    let engine = VectorSearchEngine(
      embedder: FixedEmbedder(embedding: [1.0, 0.0]),
      strategy: .indexed(candidateMultiplier: 2, batchSize: 10, maxConcurrentBatches: 1)
    )

    _ = try await engine.search(
      query: .vector([1.0, 0.0]),
      storage: storage,
      options: try SearchOptions(numResults: 1)
    )

    let snapshot = await storage.snapshot()
    #expect(snapshot.searchVectorCandidatesCalls == 1)
    #expect(snapshot.loadDocumentsCalled == true)
    #expect(snapshot.loadDocumentsIdsCalls == 1)
    #expect(Set(snapshot.lastLoadedIds) == Set([doc1.id, doc2.id]))
  }
}
