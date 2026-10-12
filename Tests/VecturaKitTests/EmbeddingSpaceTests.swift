import Foundation
import Testing
@testable import VecturaKit

@Suite("Embedding space persistence")
struct EmbeddingSpaceTests {
  private struct TaggedEmbedder: VecturaEmbedder {
    let embeddingSpaceIdentifier: String?
    var dimension: Int { get async throws { 2 } }
    func embed(texts: [String]) async throws -> [[Float]] { Array(repeating: [1, 0], count: texts.count) }
  }

  @Test("Existing providers retain default query behavior and untagged persistence")
  func legacyDefaults() async throws {
    let embedder = DeterministicEmbedder(dimension: 2)
    #expect(embedder.embeddingSpaceIdentifier == nil)
    #expect(try await embedder.embedQuery(text: "query") == embedder.embed(text: "query"))
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let config = try VecturaConfig(name: "legacy", directoryURL: directory)
    let database = try await VecturaKit(config: config, embedder: embedder)
    _ = try await database.addDocument(text: "existing")
    let reopened = try await VecturaKit(config: config, embedder: embedder)
    #expect(try await reopened.documentCount == 1)
    #expect(!FileManager.default.fileExists(
      atPath: directory.appendingPathComponent("legacy/.vectura-embedding-space").path
    ))
  }

  @Test("Matching identity reopens, changed model/quant/dimension/prompt identities fail",
    arguments: ["model2:q8:128:search", "model1:bf16:128:search", "model1:q8:256:search", "model1:q8:128:other"])
  func incompatibleIdentities(_ changed: String) async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let config = try VecturaConfig(name: "tagged", directoryURL: directory)
    let embedder = TaggedEmbedder(embeddingSpaceIdentifier: "model1:q8:128:search")
    let database = try await VecturaKit(config: config, embedder: embedder)
    _ = try await database.addDocument(text: "document")
    let reopened = try await VecturaKit(config: config, embedder: embedder)
    #expect(try await reopened.documentCount == 1) // Metadata is not counted as a document.
    await #expect(throws: VecturaError.self) {
      try await VecturaKit(config: config, embedder: TaggedEmbedder(embeddingSpaceIdentifier: changed))
    }
    await #expect(throws: VecturaError.self) {
      try await VecturaKit(config: config, embedder: TaggedEmbedder(embeddingSpaceIdentifier: nil))
    }
    // Reset clears documents but preserves the declared space.
    try await database.reset()
    #expect(try await database.documentCount == 0)
    let afterReset = try await VecturaKit(config: config, embedder: embedder)
    #expect(try await afterReset.documentCount == 0)
  }

  @Test("Does not silently claim an untagged database containing existing vectors")
  func legacyMigration() async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let config = try VecturaConfig(name: "legacy", directoryURL: directory)
    let database = try await VecturaKit(config: config, embedder: TaggedEmbedder(embeddingSpaceIdentifier: nil))
    _ = try await database.addDocument(text: "legacy document")
    await #expect(throws: VecturaError.self) {
      try await VecturaKit(config: config, embedder: TaggedEmbedder(embeddingSpaceIdentifier: "new-model"))
    }
    #expect(try await database.documentCount == 1)
  }

  @Test("Tagged providers reject a contradictory configured dimension at initialization")
  func dimensionMismatch() async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let config = try VecturaConfig(name: "test", directoryURL: directory, dimension: 3)
    do {
      _ = try await VecturaKit(config: config, embedder: TaggedEmbedder(embeddingSpaceIdentifier: "model"))
      Issue.record("Expected dimensionMismatch")
    } catch VecturaError.dimensionMismatch(let expected, let got) {
      #expect(expected == 2)
      #expect(got == 3)
    }
  }

  @Test("Concurrent initializers cannot overwrite a claimed identity")
  func concurrentIdentityClaims() async throws {
    for _ in 0..<20 {
      let directory = temporaryDirectory()
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: directory) }
      let successes = await withTaskGroup(of: Bool.self) { group in
        for identifier in ["model-a", "model-b"] {
          group.addTask {
            do {
              _ = try FileStorageProvider(storageDirectory: directory, embeddingSpaceIdentifier: identifier)
              return true
            } catch { return false }
          }
        }
        var count = 0
        for await success in group where success { count += 1 }
        return count
      }
      #expect(successes == 1)
      let identity = try String(
        contentsOf: directory.appendingPathComponent(".vectura-embedding-space"), encoding: .utf8
      )
      #expect(["model-a", "model-b"].contains(identity))
    }
  }

  private func temporaryDirectory() -> URL {
    URL(filePath: NSTemporaryDirectory()).appendingPathComponent("EmbeddingSpace-" + UUID().uuidString)
  }
}
