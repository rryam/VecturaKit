import Foundation
import Testing
@testable import VecturaNLKit
@testable import VecturaKit

/// Tests for VecturaKit with NLContextualEmbedding functionality
///
/// Note: These tests require macOS 14.4+, iOS 17.4+, or equivalent platform versions
/// where NLContextualEmbedding is available.
///
/// Run tests with: swift test --filter VecturaNLKitTests
@Suite("VecturaNLKit")
struct VecturaNLKitTests {

  private var shouldRunNLTests: Bool {
    if #available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *) {
      return true
    } else {
      return false
    }
  }

  private var defaultSearchOptions: VecturaConfig.SearchOptions {
    .init(
      defaultNumResults: 10,
      minThreshold: 0,
      hybridWeight: 0.5,
      k1: 1.2,
      b: 0.75
    )
  }

  private func makeTestDirectory() throws -> URL {
    let temp = FileManager.default.temporaryDirectory
    let directory = temp.appendingPathComponent(
      "VecturaNLKitTests-\(UUID().uuidString)",
      isDirectory: true
    )

    if FileManager.default.fileExists(atPath: directory.path(percentEncoded: false)) {
      try FileManager.default.removeItem(at: directory)
    }

    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )

    return directory
  }

  private func makeConfig(
    name: String = UUID().uuidString,
    directoryURL: URL,
    dimension: Int? = nil
  ) throws -> VecturaConfig {
    try VecturaConfig(
      name: name,
      directoryURL: directoryURL,
      dimension: dimension,
      searchOptions: defaultSearchOptions
    )
  }

  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  private func createVecturaKit(config: VecturaConfig) async throws -> VecturaKit? {
    guard shouldRunNLTests else {
      return nil
    }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      return try await VecturaKit(config: config, embedder: embedder)
    } catch {
      return nil
    }
  }

  @Test("Initialize NLContextualEmbedder")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func initializeEmbedder() async throws {
    guard shouldRunNLTests else { return }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      let dimension = try await embedder.dimension
      #expect(dimension > 0, "Embedding dimension should be greater than 0")
    } catch {
      return
    }
  }

  @Test("Embed single text")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func embedSingleText() async throws {
    guard shouldRunNLTests else { return }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      let text = "Hello world"
      let embedding = try await embedder.embed(text: text)
      #expect(!embedding.isEmpty, "Embedding should not be empty")

      let dimension = try await embedder.dimension
      #expect(embedding.count == dimension, "Embedding count should match dimension")
    } catch {
      return
    }
  }

  @Test("Embed multiple texts")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func embedMultipleTexts() async throws {
    guard shouldRunNLTests else { return }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      let texts = ["Hello", "World", "Swift"]
      let embeddings = try await embedder.embed(texts: texts)
      #expect(embeddings.count == 3, "Should return 3 embeddings")

      let dimension = try await embedder.dimension
      for embedding in embeddings {
        #expect(embedding.count == dimension, "Each embedding should match dimension")
      }
    } catch {
      return
    }
  }

  @Test("Add and search documents")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func addAndSearch() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let text = "Hello world"
    let ids = try await kit.addDocuments(texts: [text])
    #expect(ids.count == 1, "Should add exactly one document.")

    let results = try await kit.search(query: .text(text))
    #expect(results.count == 1, "The search should return one result after adding one document.")
    #expect(results.first?.text == text, "The text of the returned document should match the added text.")
  }

  @Test("Delete documents")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func deleteDocuments() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let text = "Delete me"
    let ids = try await kit.addDocuments(texts: [text])
    #expect(ids.count == 1, "Should add exactly one document.")

    try await kit.deleteDocuments(ids: ids)

    let results = try await kit.search(query: .text(text))
    #expect(results.isEmpty, "After deletion, the document should not be returned in search results.")
  }

  @Test("Update document")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func updateDocument() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let originalText = "Original text"
    let updatedText = "Updated text"
    let ids = try await kit.addDocuments(texts: [originalText])
    #expect(ids.count == 1, "Should add exactly one document.")

    let documentID = try #require(ids.first)
    try await kit.updateDocument(id: documentID, newText: updatedText)

    let results = try await kit.search(query: .text(updatedText))
    #expect(results.count == 1, "One document should be returned after update.")
    #expect(results.first?.text == updatedText, "The document text should be updated in the search results.")
  }

  @Test("Reset removes documents")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func reset() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    _ = try await kit.addDocuments(texts: ["Doc1", "Doc2"])
    try await kit.reset()

    let results = try await kit.search(query: "Doc")
    #expect(results.isEmpty, "After a reset, search should return no results.")
  }

  @Test("Search multiple documents")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func searchMultipleDocuments() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestNLDB", directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let texts = [
      "The quick brown fox jumps over the lazy dog",
      "A fast brown fox leaps over lazy hounds",
      "An agile brown fox",
      "Lazy dogs sleep all day",
      "Quick and nimble foxes"
    ]
    _ = try await kit.addDocuments(texts: texts)

    let results = try await kit.search(query: "brown fox")
    #expect(results.count >= 2, "Should return at least two documents related to 'brown fox'.")

    for index in 1..<results.count {
      #expect(
        results[index - 1].score >= results[index].score,
        "Search results are not sorted in descending order by score."
      )
    }
  }

  @Test("Search result limiting")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func searchNumResultsLimiting() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestNLDB", directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let texts = [
      "Document one about testing",
      "Document two about testing",
      "Document three about testing",
      "Document four about testing",
      "Document five about testing"
    ]
    _ = try await kit.addDocuments(texts: texts)

    let results = try await kit.search(query: "testing", numResults: 3)
    #expect(results.count == 3, "Should limit the search results to exactly 3 documents.")
  }

  @Test("Search high threshold")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func searchWithHighThreshold() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestNLDB", directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    let texts = [
      "Apple pie recipe",
      "Delicious apple tart",
      "Banana bread instructions"
    ]
    _ = try await kit.addDocuments(texts: texts)

    let highThreshold: Float = 0.99
    let results = try await kit.search(query: "apple", threshold: highThreshold)

    for result in results {
      #expect(
        result.score >= highThreshold,
        "Result score \(result.score) is below the high threshold \(highThreshold)."
      )
    }
  }

  @Test("Search no matches")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func searchNoMatches() async throws {
    guard shouldRunNLTests else { return }

    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestNLDB", directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    _ = try await kit.addDocuments(texts: ["Some random content"])

    let results = try await kit.search(query: "completely different query text", threshold: 0.9)
    #expect(results.isEmpty, "Search should return no results when the query does not match any document.")
  }

  @Test("Empty text error handling")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func emptyTextError() async throws {
    guard shouldRunNLTests else { return }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      _ = try await embedder.embed(text: "")
      Issue.record("Should throw error for empty text")
    } catch {
      #expect(error is NLContextualEmbedderError, "Should throw NLContextualEmbedderError")
    }
  }

  @Test("Model info property")
  @available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
  func modelInfo() async throws {
    guard shouldRunNLTests else { return }

    do {
      let embedder = try await NLContextualEmbedder(language: .english)
      let info = await embedder.modelInfo
      #expect(info.language == .english, "Language should match initialization parameter")

      _ = try await embedder.dimension
      let infoWithDimension = await embedder.modelInfo
      #expect(infoWithDimension.dimension != nil, "Dimension should be cached after first access")
    } catch {
      return
    }
  }
}
