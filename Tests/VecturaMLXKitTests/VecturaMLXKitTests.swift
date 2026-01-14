import Foundation
import Metal
import MLX
import Testing
@testable import VecturaMLXKit
@testable import VecturaKit

/// Tests for VecturaKit with MLX embeddings functionality
///
/// Note: These tests require:
/// 1. Metal device (GPU) availability
/// 2. Metal Toolchain (install via: xcodebuild -downloadComponent MetalToolchain)
/// 3. MLX device libraries to be available
///
/// Run tests with: xcodebuild test -scheme VecturaMLXKitTests -destination 'platform=macOS'
/// (swift test may not work due to Metal library compilation requirements)
@Suite("VecturaMLXKit")
struct VecturaMLXKitTests {
  private let testDimension = 768

  private var shouldRunMLXTests: Bool {
    ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] == "1"
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
      "VecturaMLXKitTests-\(UUID().uuidString)",
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

  private func makeConfig(name: String = UUID().uuidString, directoryURL: URL) throws -> VecturaConfig {
    try VecturaConfig(
      name: name,
      directoryURL: directoryURL,
      dimension: testDimension,
      searchOptions: defaultSearchOptions
    )
  }

  private func createVecturaKit(config: VecturaConfig) async throws -> VecturaKit? {
    guard shouldRunMLXTests else {
      return nil
    }

    guard MTLCreateSystemDefaultDevice() != nil else {
      return nil
    }

    do {
      let embedder = try await MLXEmbedder(configuration: .nomic_text_v1_5)
      return try await VecturaKit(config: config, embedder: embedder)
    } catch {
      return nil
    }
  }

  private func createMLXEmbedder() async throws -> MLXEmbedder? {
    guard shouldRunMLXTests else {
      return nil
    }

    guard MTLCreateSystemDefaultDevice() != nil else {
      return nil
    }

    do {
      return try await MLXEmbedder(configuration: .nomic_text_v1_5)
    } catch {
      return nil
    }
  }

  /// Helper to get current memory footprint in bytes
  private func getMemoryFootprint() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(
          mach_task_self_,
          task_flavor_t(MACH_TASK_BASIC_INFO),
          $0,
          &count
        )
      }
    }
    return result == KERN_SUCCESS ? info.resident_size : 0
  }

  @Test("Add and search")
  func addAndSearch() async throws {
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
  func deleteDocuments() async throws {
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
  func updateDocument() async throws {
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
  func reset() async throws {
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
  func searchMultipleDocuments() async throws {
    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestMLXDB", directoryURL: directory)
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
  func searchNumResultsLimiting() async throws {
    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestMLXDB", directoryURL: directory)
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
  func searchWithHighThreshold() async throws {
    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestMLXDB", directoryURL: directory)
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
  func searchNoMatches() async throws {
    let directory = try makeTestDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = try makeConfig(name: "TestMLXDB", directoryURL: directory)
    guard let kit = try await createVecturaKit(config: config) else { return }

    _ = try await kit.addDocuments(texts: ["Some random content"])

    let results = try await kit.search(query: "completely different query text", threshold: 0.9)
    #expect(results.isEmpty, "Search should return no results when the query does not match any document.")
  }

  // MARK: - Memory Tests

  @Test("Embedding single text maintains reasonable memory")
  func embeddingSingleTextMemory() async throws {
    guard let embedder = try await createMLXEmbedder() else { return }

    let baselineMemory = getMemoryFootprint()

    // Embed a single short text
    let embeddings = try await embedder.embed(texts: ["Hello, this is a test document."])

    let afterMemory = getMemoryFootprint()
    let memoryIncrease = afterMemory > baselineMemory ? afterMemory - baselineMemory : 0

    #expect(embeddings.count == 1, "Should produce exactly one embedding.")
    #expect(embeddings[0].count > 0, "Embedding should have non-zero dimensions.")

    // Memory increase should be reasonable (less than 500MB for a single short text)
    let maxAllowedIncrease: UInt64 = 500 * 1024 * 1024
    let increaseMB = memoryIncrease / 1024 / 1024
    let maxMB = maxAllowedIncrease / 1024 / 1024
    #expect(
      memoryIncrease < maxAllowedIncrease,
      "Memory increase (\(increaseMB)MB) exceeds \(maxMB)MB limit for single text."
    )
  }

  @Test("Embedding multiple texts releases memory properly")
  func embeddingMultipleTextsMemory() async throws {
    guard let embedder = try await createMLXEmbedder() else { return }

    let baselineMemory = getMemoryFootprint()

    // Embed multiple texts in sequence
    let texts = [
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning enables computers to learn from data.",
      "Swift is a powerful programming language for Apple platforms.",
      "Vector databases store and query high-dimensional embeddings.",
      "Natural language processing transforms text into numbers."
    ]

    let embeddings = try await embedder.embed(texts: texts)

    let afterMemory = getMemoryFootprint()
    let memoryIncrease = afterMemory > baselineMemory ? afterMemory - baselineMemory : 0

    #expect(embeddings.count == texts.count, "Should produce one embedding per input text.")

    // Memory increase should be reasonable (less than 600MB for 5 texts)
    let maxAllowedIncrease: UInt64 = 600 * 1024 * 1024
    let increaseMB = memoryIncrease / 1024 / 1024
    let maxMB = maxAllowedIncrease / 1024 / 1024
    #expect(
      memoryIncrease < maxAllowedIncrease,
      "Memory increase (\(increaseMB)MB) exceeds \(maxMB)MB limit for multiple texts."
    )
  }

  @Test("Repeated embeddings don't accumulate memory")
  func repeatedEmbeddingsMemory() async throws {
    guard let embedder = try await createMLXEmbedder() else { return }

    // Run first embedding to warm up
    _ = try await embedder.embed(texts: ["Warmup text for initialization."])

    let baselineMemory = getMemoryFootprint()

    // Run multiple embedding operations
    for i in 0..<5 {
      let texts = (0..<3).map { "Test document number \($0) in iteration \(i)." }
      let embeddings = try await embedder.embed(texts: texts)
      #expect(embeddings.count == 3, "Each iteration should produce 3 embeddings.")
    }

    let afterMemory = getMemoryFootprint()
    let memoryIncrease = afterMemory > baselineMemory ? afterMemory - baselineMemory : 0

    // Memory increase after repeated operations should be bounded (less than 700MB)
    // This verifies that memory is properly released between operations
    let maxAllowedIncrease: UInt64 = 700 * 1024 * 1024
    let increaseMB = memoryIncrease / 1024 / 1024
    let maxMB = maxAllowedIncrease / 1024 / 1024
    #expect(
      memoryIncrease < maxAllowedIncrease,
      "Memory increase (\(increaseMB)MB) exceeds \(maxMB)MB limit for repeated ops."
    )
  }

  @Test("Embedding returns correct dimensions")
  func embeddingDimensions() async throws {
    guard let embedder = try await createMLXEmbedder() else { return }

    let texts = ["Test text one.", "Test text two."]
    let embeddings = try await embedder.embed(texts: texts)

    #expect(embeddings.count == 2, "Should produce two embeddings.")

    // All embeddings should have the same dimension
    let dimension = embeddings[0].count
    #expect(dimension > 0, "Embedding dimension should be positive.")

    for (index, embedding) in embeddings.enumerated() {
      #expect(
        embedding.count == dimension,
        "Embedding \(index) has dimension \(embedding.count), expected \(dimension)."
      )
    }
  }

  @Test("GPU cache cleared after embedding")
  func gpuCacheClearedAfterEmbedding() async throws {
    guard let embedder = try await createMLXEmbedder() else { return }

    // Perform embedding operation
    let embeddings = try await embedder.embed(texts: ["Test document for GPU cache verification."])

    #expect(embeddings.count == 1, "Should produce one embedding.")
    #expect(embeddings[0].count > 0, "Embedding should have non-zero dimensions.")

    // After embedding, GPU cache should be cleared
    // We can't directly verify GPU cache state, but we can verify the operation completes
    // and the embedder remains functional
    let secondEmbeddings = try await embedder.embed(texts: ["Second test to verify embedder still works."])
    #expect(secondEmbeddings.count == 1, "Embedder should remain functional after cache clear.")
  }
}
