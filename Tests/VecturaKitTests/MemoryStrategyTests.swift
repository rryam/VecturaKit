import Foundation
import Testing
@testable import VecturaKit

@Suite("Memory Strategy Tests")
struct MemoryStrategyTests {

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("MemoryStrategyTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    let cleanup = {
      _ = try? FileManager.default.removeItem(at: directory)
    }
    return (directory, cleanup)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func makeEmbedder(modelSource: VecturaModelSource = .default) -> SwiftEmbedder {
    SwiftEmbedder(modelSource: modelSource)
  }

  @Test("Default strategy is automatic")
  func defaultStrategy() throws {
    let config = try VecturaConfig(name: "test-db")
    if case .automatic = config.memoryStrategy {
      // Expected
    } else {
      Issue.record("Expected automatic strategy")
    }
  }

  @Test("Full memory strategy can be set")
  func fullMemoryStrategy() throws {
    let config = try VecturaConfig(name: "test-db", memoryStrategy: .fullMemory)
    #expect(config.memoryStrategy == .fullMemory)
  }

  @Test("Indexed strategy can be set with defaults")
  func indexedStrategyDefaults() throws {
    let config = try VecturaConfig(name: "test-db", memoryStrategy: .indexed())

    if case .indexed(let multiplier, let batch, let maxConcurrent) = config.memoryStrategy {
      #expect(multiplier == VecturaConfig.MemoryStrategy.defaultCandidateMultiplier)
      #expect(batch == VecturaConfig.MemoryStrategy.defaultBatchSize)
      #expect(maxConcurrent == VecturaConfig.MemoryStrategy.defaultMaxConcurrentBatches)
    } else {
      Issue.record("Expected indexed strategy")
    }
  }

  @Test("Indexed strategy can be customized")
  func indexedStrategyCustom() throws {
    let config = try VecturaConfig(
      name: "test-db",
      memoryStrategy: .indexed(
        candidateMultiplier: 5,
        batchSize: 50,
        maxConcurrentBatches: 2
      )
    )

    if case .indexed(let multiplier, let batch, let maxConcurrent) = config.memoryStrategy {
      #expect(multiplier == 5)
      #expect(batch == 50)
      #expect(maxConcurrent == 2)
    } else {
      Issue.record("Expected indexed strategy")
    }
  }

  @Test("VecturaKit initializes with automatic strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func initializesWithAutomaticStrategy() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    #expect(try await vectura.documentCount == 0)
  }

  @Test("VecturaKit initializes with full memory strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func initializesWithFullMemoryStrategy() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .fullMemory
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    #expect(try await vectura.documentCount == 0)
  }

  @Test("VecturaKit initializes with indexed strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func initializesWithIndexedStrategy() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .indexed()
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    #expect(try await vectura.documentCount == 0)
  }

  @Test("Search works with automatic strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func searchWithAutomaticStrategy() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let texts = ["Machine learning", "Deep learning", "Neural networks"]
    _ = try await vectura.addDocuments(texts: texts)

    let results = try await vectura.search(query: "learning")
    #expect(results.count > 0)
  }

  @Test("Search works with full memory strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func searchWithFullMemoryStrategy() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .fullMemory
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let texts = ["Machine learning", "Deep learning", "Neural networks"]
    _ = try await vectura.addDocuments(texts: texts)

    let results = try await vectura.search(query: "learning")
    #expect(results.count > 0)
  }

  @Test("Search works with indexed strategy and FileStorageProvider")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func searchWithIndexedStrategyFileStorage() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .indexed()
    )

    // FileStorageProvider doesn't support IndexedVecturaStorage,
    // so it should automatically fall back to full memory mode
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let texts = ["Machine learning", "Deep learning", "Neural networks"]
    _ = try await vectura.addDocuments(texts: texts)

    let results = try await vectura.search(query: "learning")
    #expect(results.count > 0)
  }

  @Test("Memory strategy is backward compatible")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func backwardCompatibility() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    // Old code without specifying memory strategy
    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory
    )

    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let texts = ["Document 1", "Document 2", "Document 3"]
    let ids = try await vectura.addDocuments(texts: texts)

    let results = try await vectura.search(query: "Document")
    #expect(results.count == 3)
    #expect(ids.contains(results[0].id))
  }
}

@Suite("Mock Indexed Storage Tests")
struct MockIndexedStorageTests {

  /// Mock implementation of IndexedVecturaStorage for testing
  actor MockIndexedStorage: IndexedVecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]

    func createStorageDirectoryIfNeeded() async throws {
      // No-op for mock
    }

    func loadDocuments() async throws -> [VecturaDocument] {
      return Array(documents.values)
    }

    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument] {
      let docs = Array(documents.values)
      let start = min(offset, docs.count)
      let end = min(offset + limit, docs.count)
      return Array(docs[start..<end])
    }

    func getTotalDocumentCount() async throws -> Int {
      return documents.count
    }

    func saveDocument(_ document: VecturaDocument) async throws {
      documents[document.id] = document
    }

    func deleteDocument(withID id: UUID) async throws {
      documents[id] = nil
    }

    func updateDocument(_ document: VecturaDocument) async throws {
      documents[document.id] = document
    }

    func searchVectorCandidates(
      queryEmbedding: [Float],
      topK: Int,
      prefilterSize: Int
    ) async throws -> [UUID]? {
      // Simple mock: return all document IDs
      return Array(documents.keys)
    }

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
      var result: [UUID: VecturaDocument] = [:]
      for id in ids {
        if let doc = documents[id] {
          result[id] = doc
        }
      }
      return result
    }
  }

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("MockIndexedStorageTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    let cleanup = {
      _ = try? FileManager.default.removeItem(at: directory)
    }
    return (directory, cleanup)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func makeEmbedder(modelSource: VecturaModelSource = .default) -> SwiftEmbedder {
    SwiftEmbedder(modelSource: modelSource)
  }

  @Test("VecturaKit works with mock indexed storage")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func mockIndexedStorage() async throws {

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .indexed()
    )

    let mockStorage = MockIndexedStorage()
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: mockStorage)

    let texts = ["Machine learning", "Deep learning", "Neural networks"]
    _ = try await vectura.addDocuments(texts: texts)

    #expect(try await vectura.documentCount == 3)

    let results = try await vectura.search(query: "learning")
    #expect(results.count > 0)
  }

  @Test("Mock indexed storage supports pagination")
  func mockPagination() async throws {
    let mockStorage = MockIndexedStorage()

    // Add test documents
    for i in 0..<100 {
      let doc = VecturaDocument(
        id: UUID(),
        text: "Document \(i)",
        embedding: Array(repeating: Float(i), count: 384)
      )
      try await mockStorage.saveDocument(doc)
    }

    let count = try await mockStorage.getTotalDocumentCount()
    #expect(count == 100)

    let page1 = try await mockStorage.loadDocuments(offset: 0, limit: 10)
    #expect(page1.count == 10)

    let page2 = try await mockStorage.loadDocuments(offset: 10, limit: 10)
    #expect(page2.count == 10)

    // Verify pages are different
    let page1Ids = Set(page1.map { $0.id })
    let page2Ids = Set(page2.map { $0.id })
    #expect(page1Ids.isDisjoint(with: page2Ids))
  }

  @Test("Mock indexed storage handles batch loading with custom parameters")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func batchLoadingWithCustomParams() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .indexed(
        candidateMultiplier: 10,
        batchSize: 5,
        maxConcurrentBatches: 2
      )
    )

    let mockStorage = MockIndexedStorage()
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: mockStorage)

    // Add enough documents to trigger batching
    var texts: [String] = []
    for i in 0..<20 {
      texts.append("Document \(i) about machine learning")
    }
    _ = try await vectura.addDocuments(texts: texts)

    #expect(try await vectura.documentCount == 20)

    let results = try await vectura.search(query: "machine learning")
    #expect(results.count > 0)
  }
}

@Suite("Partial Failure Handling Tests")
struct PartialFailureTests {

  /// Mock storage that simulates failures for specific batches
  actor FailingMockStorage: IndexedVecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]
    private var failingIds: Set<UUID> = []

    func setFailingIds(_ ids: Set<UUID>) {
      failingIds = ids
    }

    func createStorageDirectoryIfNeeded() async throws {
      // No-op for mock
    }

    func loadDocuments() async throws -> [VecturaDocument] {
      return Array(documents.values)
    }

    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument] {
      let docs = Array(documents.values)
      let start = min(offset, docs.count)
      let end = min(offset + limit, docs.count)
      return Array(docs[start..<end])
    }

    func getTotalDocumentCount() async throws -> Int {
      return documents.count
    }

    func saveDocument(_ document: VecturaDocument) async throws {
      documents[document.id] = document
    }

    func deleteDocument(withID id: UUID) async throws {
      documents[id] = nil
    }

    func updateDocument(_ document: VecturaDocument) async throws {
      documents[document.id] = document
    }

    func searchVectorCandidates(
      queryEmbedding: [Float],
      topK: Int,
      prefilterSize: Int
    ) async throws -> [UUID]? {
      return Array(documents.keys)
    }

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
      // Simulate failure for specific IDs
      for id in ids where failingIds.contains(id) {
        throw VecturaError.loadFailed("Simulated failure for batch containing \(id)")
      }

      var result: [UUID: VecturaDocument] = [:]
      for id in ids {
        if let doc = documents[id] {
          result[id] = doc
        }
      }
      return result
    }
  }

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("PartialFailureTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    let cleanup = {
      _ = try? FileManager.default.removeItem(at: directory)
    }
    return (directory, cleanup)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func makeEmbedder(modelSource: VecturaModelSource = .default) -> SwiftEmbedder {
    SwiftEmbedder(modelSource: modelSource)
  }

  @Test("Search succeeds with partial batch failures")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func partialBatchFailure() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-db",
      directoryURL: directory,
      memoryStrategy: .indexed(
        candidateMultiplier: 5,
        batchSize: 3,  // Small batch to force multiple batches
        maxConcurrentBatches: 2
      )
    )

    let mockStorage = FailingMockStorage()
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: mockStorage)

    // Add documents
    let texts = [
      "AI and robotics",              // Batch 0 (will fail)
      "Computer vision techniques",   // Batch 0 (will fail)
      "Neural network architecture",  // Batch 0 (will fail)
      "Machine learning basics",      // Batch 1 (will succeed)
      "Deep learning fundamentals"    // Batch 1 (will succeed)
    ]
    let ids = try await vectura.addDocuments(texts: texts)

    // Configure storage to fail for one document (simulating one batch failure)
    // With batchSize=3, documents will be split into batches:
    // - Batch 0: ids[0], ids[1], ids[2] (no "learning" matches)
    // - Batch 1: ids[3], ids[4] (contains "learning" matches)
    // Setting ids[0] to fail should cause Batch 0 to fail
    await mockStorage.setFailingIds(Set([ids[0]]))

    // Search should still work with remaining documents from successful batches
    let results = try await vectura.search(query: "learning")

    // We should get results from the documents that didn't fail
    // Batch 1 contains documents matching "learning", so we expect results
    #expect(results.count > 0, "Should get results from successful batches")

    // Verify that the failed document is not in results
    let resultIds = Set(results.map { $0.id })
    #expect(!resultIds.contains(ids[0]), "Failed document should not appear in results")

    // Verify that results come from the successful batch (Batch 1)
    let successfulBatchIds = Set([ids[3], ids[4]])
    let hasSuccessfulResults = resultIds.intersection(successfulBatchIds).count > 0
    #expect(hasSuccessfulResults, "Should have at least one result from successful batch")
  }
}
