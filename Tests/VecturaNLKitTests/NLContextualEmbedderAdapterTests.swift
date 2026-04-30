import Foundation
import Testing
@testable import VecturaKit
@testable import VecturaNLKit

@Suite("NLContextualEmbedder Adapter")
struct NLContextualEmbedderAdapterTests {
  private func makeDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("NLContextualEmbedderAdapterTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    return directory
  }

  @Test("Batch embedding rejects empty element")
  func batchEmbeddingRejectsEmptyElement() async throws {
    let embedder = try await NLContextualEmbedder(language: .english)

    await #expect(throws: NLContextualEmbedderError.self) {
      _ = try await embedder.embed(texts: ["valid", ""])
    }
  }

  @Test("Vector queries work with NL embeddings")
  func vectorQueriesWork() async throws {
    let directory = try makeDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let embedder = try await NLContextualEmbedder(language: .english)
    let config = try VecturaConfig(name: "nl-adapter-db", directoryURL: directory)
    let kit = try await VecturaKit(config: config, embedder: embedder)

    _ = try await kit.addDocuments(texts: [
      "SwiftUI powers user interfaces on Apple platforms.",
      "Relational databases use tables, indexes, and transactions.",
      "Vector search finds semantically similar content.",
    ])

    let queryEmbedding = try await embedder.embed(text: "Apple app development")
    let results = try await kit.search(query: .vector(queryEmbedding), numResults: 2)

    let first = try #require(results.first)
    #expect(first.text.localizedCaseInsensitiveContains("Apple"))
  }
}
