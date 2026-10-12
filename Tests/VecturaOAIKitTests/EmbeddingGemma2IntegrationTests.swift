import Foundation
import Testing
import VecturaKit
import VecturaOAIKit

@Suite("EmbeddingGemma2 real llama.cpp inference", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["VECTURA_GEMMA2_INTEGRATION"] == "1"))
struct EmbeddingGemma2IntegrationTests {
  @Test("Real image query retrieves stored text without adding image documents",
    .enabled(if: ProcessInfo.processInfo.environment["VECTURA_GEMMA2_IMAGE_INTEGRATION"] == "1"))
  func imageToTextInference() async throws {
    let environment = ProcessInfo.processInfo.environment
    let precision = try #require(EmbeddingGemma2Configuration.Precision(
      rawValue: environment["VECTURA_GEMMA2_PRECISION"] ?? "Q8_0"
    ))
    let config = try EmbeddingGemma2Configuration(precision: precision, enableImageQueries: true)
    let embedder = try EmbeddingGemma2Embedder(
      configuration: config, baseURL: environment["VECTURA_GEMMA2_BASE_URL"] ?? "http://127.0.0.1:8080/v1"
    )
    // Owned deterministic 64x64 RGB PNG controls, generated with zlib (no external media).
    let red = try #require(Data(base64Encoded: """
      iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I
      8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc
      8OLDQitxAAAAAElFTkSuQmCC
      """, options: .ignoreUnknownCharacters))
    let blue = try #require(Data(base64Encoded: """
      iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdNvJ8QUNyPEFDcjxBQ3I
      8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ2oPcf8
      8OIhvJ6vAAAAAElFTkSuQmCC
      """, options: .ignoreUnknownCharacters))
    let redVector = try await embedder.embedImageQuery(data: red, format: .png)
    let blueVector = try await embedder.embedImageQuery(data: blue, format: .png)
    let repeated = try await embedder.embedImageQuery(data: red, format: .png)
    #expect(redVector.count == 768)
    #expect(abs(redVector.reduce(0) { $0 + $1 * $1 } - 1) < 0.0001)
    #expect(zip(redVector, repeated).allSatisfy { abs($0 - $1) < 0.002 })
    // Detect an implementation that silently embeds a marker/caption instead of image content.
    #expect(zip(redVector, blueVector).contains { abs($0 - $1) > 0.001 })
    let directory = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let databaseConfig = try VecturaConfig(name: "image-query", directoryURL: directory,
      searchOptions: .init(minThreshold: nil, hybridWeight: 1))
    // Index with a text-only configuration, then reopen with image queries enabled.
    let textEmbedder = try EmbeddingGemma2Embedder(configuration: .init(precision: precision),
      baseURL: environment["VECTURA_GEMMA2_BASE_URL"] ?? "http://127.0.0.1:8080/v1")
    let textDatabase = try await VecturaKit(config: databaseConfig, embedder: textEmbedder)
    let texts = ["A solid red square.", "A solid blue square.", "A bright green triangle."]
    let identifiers = try await textDatabase.addDocuments(texts: texts)
    if let baselinePath = environment["VECTURA_GEMMA2_TEXT_BASELINE"] {
      let baseline = try JSONDecoder().decode([[Float]].self, from: Data(contentsOf: URL(filePath: baselinePath)))
      let current = try await embedder.embed(texts: texts)
      #expect(baseline.count == current.count)
      for (before, after) in zip(baseline, current) {
        #expect(before.count == after.count)
        #expect(zip(before, after).allSatisfy { abs($0 - $1) < 0.002 })
      }
    }
    let database = try await VecturaKit(config: databaseConfig, embedder: embedder)
    let redResults = try await database.search(query: .vector(redVector), numResults: 3)
    let blueResults = try await database.search(query: .vector(blueVector), numResults: 3)
    #expect(redResults.first?.id == identifiers[0])
    #expect(blueResults.first?.id == identifiers[1])
    #expect(try await database.documentCount == 3)
    #expect(Set(try await database.getAllDocuments().map(\.text)) == Set(texts))
  }

  @Test("Real text inference, Matryoshka vectors, retrieval and database reopen")
  func realInference() async throws {
    let environment = ProcessInfo.processInfo.environment
    let url = environment["VECTURA_GEMMA2_BASE_URL"] ?? "http://127.0.0.1:8080/v1"
    let precision = try #require(EmbeddingGemma2Configuration.Precision(
      rawValue: environment["VECTURA_GEMMA2_PRECISION"] ?? "Q8_0"
    ))
    let documents = [
      "The northern lights are caused by charged particles from the Sun interacting with Earth's atmosphere.",
      "Sourdough bread is made by fermenting flour and water with a starter culture.",
      "The swift programming language uses actors to coordinate access to mutable state."
    ]
    for dimensions in EmbeddingGemma2Configuration.Dimensions.allCases {
      let config = try EmbeddingGemma2Configuration(precision: precision, dimensions: dimensions, batchSize: 2)
      let embedder = try EmbeddingGemma2Embedder(configuration: config, baseURL: url)
      let batch = try await embedder.embed(texts: documents)
      let single = try await embedder.embed(text: documents[0])
      #expect(batch.count == documents.count)
      #expect(zip(batch[0], single).allSatisfy { abs($0 - $1) < 0.002 })
      for vector in batch {
        #expect(vector.count == dimensions.rawValue)
        #expect(vector.allSatisfy { $0.isFinite })
        #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 0.0001)
      }
      let directory = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: directory) }
      let databaseConfig = try VecturaConfig(
        name: "gemma2", directoryURL: directory,
        searchOptions: .init(minThreshold: nil, hybridWeight: 1)
      )
      let database = try await VecturaKit(config: databaseConfig, embedder: embedder)
      let identifiers = try await database.addDocuments(texts: documents)
      let results = try await database.search(query: "What causes the aurora?", numResults: 3)
      #expect(results.first?.id == identifiers[0])
      let reopened = try await VecturaKit(config: databaseConfig, embedder: embedder)
      #expect(try await reopened.documentCount == 3)
      let persisted = try await reopened.search(query: "How do I ferment bread?", numResults: 3)
      #expect(persisted.first?.id == identifiers[1])
    }
  }
}
