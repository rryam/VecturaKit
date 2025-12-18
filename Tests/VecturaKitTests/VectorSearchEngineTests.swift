import Foundation
import Testing
@testable import VecturaKit

@Suite("VectorSearchEngine")
struct VectorSearchEngineTests {

  private struct FixedEmbedder: VecturaEmbedder {
    let dimensionValue: Int

    init(dimension: Int) {
      self.dimensionValue = dimension
    }

    var dimension: Int {
      get async throws { dimensionValue }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      Array(repeating: [Float](repeating: 0.1, count: dimensionValue), count: texts.count)
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

  @Test("Throws on malformed embedding dimensions")
  func throwsOnMalformedEmbeddingDimensions() async throws {
    let expectedDimension = 3
    let storage = FixedStorage(
      documents: [
        VecturaDocument(
          id: UUID(),
          text: "valid",
          embedding: [0.1, 0.2, 0.3]
        ),
        VecturaDocument(
          id: UUID(),
          text: "malformed",
          embedding: [0.4, 0.5]
        ),
      ]
    )

    let engine = VectorSearchEngine(
      embedder: FixedEmbedder(dimension: expectedDimension),
      strategy: .fullMemory
    )

    do {
      _ = try await engine.search(
        query: .vector([0.9, 0.8, 0.7]),
        storage: storage,
        options: try SearchOptions(numResults: 10)
      )
      Issue.record("Expected dimensionMismatch error for malformed embedding")
    } catch let error as VecturaError {
      switch error {
      case .dimensionMismatch(let expected, let got):
        #expect(expected == expectedDimension)
        #expect(got == 2)
      default:
        Issue.record("Unexpected error: \(error)")
      }
    } catch {
      Issue.record("Unexpected error type: \(error)")
    }
  }
}

