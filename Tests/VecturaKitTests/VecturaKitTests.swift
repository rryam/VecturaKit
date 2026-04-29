import Foundation
import Testing
@testable import VecturaKit

@Suite("VecturaKit")
struct VecturaKitTests {
  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("VecturaKitTests-\(UUID().uuidString)", isDirectory: true)
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

  private func makeVecturaConfig(
    name: String = "test-db-\(UUID().uuidString)",
    dimension: Int? = nil,
    searchOptions: VecturaConfig.SearchOptions = .init()
  ) throws -> (VecturaConfig, () -> Void) {
    let (directory, cleanup) = try makeTestDirectory()
    let config = try VecturaConfig(
      name: name,
      directoryURL: directory,
      dimension: dimension,
      searchOptions: searchOptions
    )
    return (config, cleanup)
  }

  private func makeEmbedder(dimension: Int = 384) -> DeterministicEmbedder {
    DeterministicEmbedder(dimension: dimension)
  }

  private struct MismatchEmbedder: VecturaEmbedder {
    let dimensionValue: Int

    init(dimension: Int = 3) {
      self.dimensionValue = dimension
    }

    var dimension: Int {
      get async throws { dimensionValue }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      let embedding = [Float](repeating: 0.1, count: dimensionValue)
      if texts.count <= 1 {
        return [embedding]
      }
      return Array(repeating: embedding, count: texts.count - 1)
    }
  }

  private struct FixedEmbedder: VecturaEmbedder {
    let embedding: [Float]

    init(embedding: [Float] = [1.0, 0.0, 0.0]) {
      self.embedding = embedding
    }

    var dimension: Int {
      get async throws { embedding.count }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      Array(repeating: embedding, count: texts.count)
    }
  }

  @Test("Add and search document")
  func addAndSearchDocument() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let text = "This is a test document"
    let id = try await vectura.addDocument(text: text)

    let results = try await vectura.search(query: "test document")
    #expect(results.count == 1)
    #expect(results[0].id == id)
    #expect(results[0].text == text)
  }

  @Test("Add multiple documents")
  func addMultipleDocuments() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let documents = [
      "The quick brown fox jumps over the lazy dog",
      "Pack my box with five dozen liquor jugs",
      "How vexingly quick daft zebras jump"
    ]

    let ids = try await vectura.addDocuments(texts: documents)
    #expect(ids.count == 3)

    let results = try await vectura.search(query: "quick jumping animals")
    #expect(results.count >= 2)
    #expect(results[0].score > results[1].score)
  }

  @Test("Persistence across instances")
  func persistence() async throws {
    let databaseName = UUID().uuidString
    let (config, cleanup) = try makeVecturaConfig(name: "test-db-\(databaseName)")
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let texts = ["Document 1", "Document 2"]
    let ids = try await vectura.addDocuments(texts: texts)

    let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    let results = try await newVectura.search(query: "Document")
    #expect(results.count == 2)
    #expect(ids.contains(results[0].id))
    #expect(ids.contains(results[1].id))
  }

  @Test("Search threshold reduces results")
  func searchThreshold() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let documents = [
      "Very relevant document about cats",
      "Somewhat relevant about pets",
      "Completely irrelevant about weather"
    ]
    _ = try await vectura.addDocuments(texts: documents)

    let results = try await vectura.search(query: "cats and pets", threshold: 0.8)
    #expect(results.count < 3)
  }

  @Test("Custom identifiers are preserved")
  func customIds() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let customId = UUID()
    let text = "Document with custom ID"

    let resultId = try await vectura.addDocument(text: text, id: customId)
    #expect(customId == resultId)

    let results = try await vectura.search(query: .text(text))
    #expect(results.count == 1)
    #expect(results[0].id == customId)
  }

  @Test("Embedder count mismatch throws")
  func embedderCountMismatch() async throws {
    let (config, cleanup) = try makeVecturaConfig(dimension: 3)
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: MismatchEmbedder(dimension: 3))

    do {
      _ = try await vectura.addDocuments(texts: ["First", "Second"])
      Issue.record("Expected invalidInput error for mismatched embedding count")
    } catch let error as VecturaError {
      switch error {
      case .invalidInput(let reason):
        #expect(reason.contains("Embedder returned"))
      default:
        Issue.record("Unexpected error: \(error)")
      }
    } catch {
      Issue.record("Unexpected error type: \(error)")
    }
  }

  @Test("Model reuse remains performant")
  func modelReuse() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let start = Date()
    for i in 1...5 {
      _ = try await vectura.addDocument(text: "Test document \(i)")
    }
    let duration = Date().timeIntervalSince(start)

    #expect(duration < 10.0)
  }

  @Test("Empty search returns no results")
  func emptySearch() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let results = try await vectura.search(query: "test query")
    #expect(results.count == 0, "Search on empty database should return no results")
  }

  @Test("Config dimension overrides embedder dimension")
  func configDimensionOverride() async throws {
    // Use config with dimension 64, even though the test embedder has a different dimension.
    let (configWithDim, cleanup) = try makeVecturaConfig(
      name: "custom-dim-db-\(UUID().uuidString)",
      dimension: 64
    )
    defer { cleanup() }
    let vectura = try await VecturaKit(config: configWithDim, embedder: makeEmbedder())

    // When adding document, embedding from embedder will have different dimension
    // This should cause dimensionMismatch error in validateDimension
    do {
      _ = try await vectura.addDocument(text: "Test document")
      Issue.record("Expected dimension mismatch error")
    } catch let error as VecturaError {
      switch error {
      case .dimensionMismatch(let expected, let got):
        #expect(expected == 64)  // config dimension
        #expect(got != 64)  // embedder's actual dimension differs
      default:
        Issue.record("Wrong error type: \(error)")
      }
    } catch {
      Issue.record("Unexpected error type: \(error)")
    }
  }

  @Test("Duplicate identifiers overwrite documents")
  func duplicateIds() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id = UUID()
    let text1 = "First document"
    let text2 = "Second document"

    _ = try await vectura.addDocument(text: text1, id: id)
    _ = try await vectura.addDocument(text: text2, id: id)

    let results = try await vectura.search(query: .text(text2))
    #expect(results.count == 1)
    #expect(results[0].text == text2)
  }

  @Test("Threshold edge cases")
  func searchThresholdEdgeCases() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    _ = try await vectura.addDocuments(texts: ["Test document"])

    let highThresholdResults = try await vectura.search(query: "completely different query", threshold: 0.95)
    #expect(highThresholdResults.count <= 1)

    let allResults = try await vectura.search(query: "completely different", threshold: 0.0)
    #expect(allResults.count >= 0)
  }

  @Test("Large number of documents")
  func largeNumberOfDocuments() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let documentCount = 100
    let documents = (0..<documentCount).map { "Test document number \($0)" }

    let ids = try await vectura.addDocuments(texts: documents)
    #expect(ids.count == documentCount)

    let results = try await vectura.search(query: "document", numResults: 10)
    #expect(results.count == 10)
  }

  @Test("Persistence after reset")
  func persistenceAfterReset() async throws {
    let databaseName = UUID().uuidString
    let (config, cleanup) = try makeVecturaConfig(name: "test-db-\(databaseName)")
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let text = "Test document"
    _ = try await vectura.addDocument(text: text)

    try await vectura.reset()

    let results = try await vectura.search(query: .text(text))
    #expect(results.count == 0)

    let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    let newResults = try await newVectura.search(query: .text(text))
    #expect(newResults.count == 0)
  }

  @Test("Custom storage directory")
  func customStorageDirectory() async throws {
    let customDirectoryURL = URL(filePath: NSTemporaryDirectory())
      .appending(path: "VecturaKitTest-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: customDirectoryURL) }

    let databaseName = "test-\(UUID().uuidString)"
    let instance = try await VecturaKit(
      config: try .init(name: databaseName, directoryURL: customDirectoryURL),
      embedder: makeEmbedder()
    )
    let text = "Test document"
    let id = UUID()
    _ = try await instance.addDocument(text: text, id: id)

    let dbDirectory = customDirectoryURL.appendingPathComponent(databaseName, isDirectory: true)
    let documentPath = dbDirectory.appendingPathComponent("\(id).json").path

    #expect(
      FileManager.default.fileExists(atPath: documentPath),
      "Expected stored document at \(documentPath)"
    )

    // Verify persistence: create new instance and load from disk
    let newInstance = try await VecturaKit(
      config: .init(name: databaseName, directoryURL: customDirectoryURL),
      embedder: makeEmbedder()
    )
    #expect(try await newInstance.documentCount == 1, "New instance should load document from disk")

    // Verify deletion removes file from disk
    try await newInstance.deleteDocuments(ids: [id])
    #expect(
      !FileManager.default.fileExists(atPath: documentPath),
      "Document file should be deleted from disk"
    )

    // Verify third instance sees the deletion
    let thirdInstance = try await VecturaKit(
      config: .init(name: databaseName, directoryURL: customDirectoryURL),
      embedder: makeEmbedder()
    )
    #expect(try await thirdInstance.documentCount == 0, "Third instance should reflect deletion")
  }

  @Test("Custom storage provider")
  func customStorageProvider() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }

    // Create a custom in-memory storage provider
    let customStorage = InMemoryStorageProvider()
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: customStorage)

    // Add documents
    let texts = ["Custom storage document 1", "Custom storage document 2"]
    let ids = try await vectura.addDocuments(texts: texts)
    #expect(ids.count == 2)

    // Verify documents were stored in custom provider
    #expect(await customStorage.documentCount == 2)

    // Search
    let results = try await vectura.search(query: "Custom storage")
    #expect(results.count == 2)

    // Delete one document
    try await vectura.deleteDocuments(ids: [ids[0]])
    #expect(await customStorage.documentCount == 1)

    // Create a new instance with the same custom storage
    let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: customStorage)
    let newResults = try await newVectura.search(query: "Custom")
    #expect(newResults.count == 1)
    #expect(newResults[0].id == ids[1])
  }

  // MARK: - getDocument Tests

  @Test("getDocument returns correct document for existing ID")
  func getDocumentExistingID() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let text = "Document to retrieve by ID"
    let id = try await vectura.addDocument(text: text)

    let found = try await vectura.getDocument(id: id)
    #expect(found != nil)
    #expect(found?.id == id)
    #expect(found?.text == text)
  }

  @Test("getDocument returns nil for non-existent ID")
  func getDocumentNonExistentID() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let ghostID = UUID()
    let found = try await vectura.getDocument(id: ghostID)
    #expect(found == nil)
  }

  @Test("getDocument returns nil after document is deleted")
  func getDocumentAfterDelete() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id = try await vectura.addDocument(text: "To be deleted")
    #expect(try await vectura.getDocument(id: id) != nil)

    try await vectura.deleteDocuments(ids: [id])
    #expect(try await vectura.getDocument(id: id) == nil)
  }

  @Test("getDocument returns updated text after update")
  func getDocumentAfterUpdate() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id = try await vectura.addDocument(text: "Original text")
    try await vectura.updateDocument(id: id, newText: "Updated text")

    let found = try await vectura.getDocument(id: id)
    #expect(found?.text == "Updated text")
  }

  @Test("getDocument works with custom storage provider via default protocol implementation")
  func getDocumentCustomStorage() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }

    let customStorage = InMemoryStorageProvider()
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: customStorage)

    let text = "Custom storage document"
    let id = try await vectura.addDocument(text: text)

    let found = try await vectura.getDocument(id: id)
    #expect(found != nil)
    #expect(found?.id == id)
    #expect(found?.text == text)
  }

  // MARK: - documentExists Tests

  @Test("documentExists returns true for existing document")
  func documentExistsTrue() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id = try await vectura.addDocument(text: "Existing document")
    #expect(try await vectura.documentExists(id: id) == true)
  }

  @Test("documentExists returns false for non-existent ID")
  func documentExistsFalse() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    #expect(try await vectura.documentExists(id: UUID()) == false)
  }

  @Test("documentExists returns false after document is deleted")
  func documentExistsAfterDelete() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id = try await vectura.addDocument(text: "Will be deleted")
    #expect(try await vectura.documentExists(id: id) == true)

    try await vectura.deleteDocuments(ids: [id])
    #expect(try await vectura.documentExists(id: id) == false)
  }

  @Test("documentExists returns true immediately after add")
  func documentExistsAfterAdd() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    #expect(try await vectura.documentExists(id: UUID()) == false)
    let id = try await vectura.addDocument(text: "Newly added")
    #expect(try await vectura.documentExists(id: id) == true)
  }

  @Test("VecturaKit lookup APIs survive restart with a partially populated file cache")
  func vecturaKitLookupAfterRestartWithPartialCache() async throws {
    let databaseName = "partial-cache-\(UUID().uuidString)"
    let (config, cleanup) = try makeVecturaConfig(name: databaseName, dimension: 3)
    defer { cleanup() }

    let initialVectura = try await VecturaKit(config: config, embedder: FixedEmbedder())
    let originalID = try await initialVectura.addDocument(text: "Persisted before restart")

    let restartedVectura = try await VecturaKit(config: config, embedder: FixedEmbedder())
    let newID = try await restartedVectura.addDocument(text: "Added after restart")

    let originalDoc = try await restartedVectura.getDocument(id: originalID)
    #expect(originalDoc?.id == originalID)
    #expect(originalDoc?.text == "Persisted before restart")
    #expect(try await restartedVectura.documentExists(id: originalID) == true)
    #expect(try await restartedVectura.documentCount == 2)

    let allDocuments = try await restartedVectura.getAllDocuments()
    #expect(allDocuments.count == 2)
    #expect(Set(allDocuments.map(\.id)) == Set([originalID, newID]))
  }

  // MARK: - deleteDocuments No-Op Bug Fix Tests

  @Test("deleteDocuments with non-existent ID does not throw")
  func deleteNonExistentIDIsNoOp() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    // Should not throw even though the ID was never added
    try await vectura.deleteDocuments(ids: [UUID()])
  }

  @Test("deleteDocuments with mixed valid and non-existent IDs deletes all valid ones")
  func deleteMixedIDsDeletesAllValid() async throws {
    let (config, cleanup) = try makeVecturaConfig()
    defer { cleanup() }
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    let id1 = try await vectura.addDocument(text: "Document one")
    let id2 = try await vectura.addDocument(text: "Document two")
    let ghostID = UUID()

    // ghost ID is sandwiched between two real IDs
    try await vectura.deleteDocuments(ids: [id1, ghostID, id2])

    // Both real documents must be gone — the ghost must not block id2's deletion
    #expect(try await vectura.documentExists(id: id1) == false)
    #expect(try await vectura.documentExists(id: id2) == false)
    #expect(try await vectura.documentCount == 0)
  }

  @Test("FileStorageProvider is stateless and reads from disk")
  func fileStorageProviderStateless() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    // Create a FileStorageProvider
    let provider = try FileStorageProvider(storageDirectory: directory)

    // Initially should have no documents
    let initialDocs = try await provider.loadDocuments()
    #expect(initialDocs.count == 0)

    // Create a test document and save it
    let doc1 = VecturaDocument(
      id: UUID(),
      text: "Test document 1",
      embedding: [1.0, 2.0, 3.0]
    )
    try await provider.saveDocument(doc1)

    // Verify the file was written to disk
    let fileURL = directory.appendingPathComponent("\(doc1.id).json")
    #expect(FileManager.default.fileExists(atPath: fileURL.path))

    // Load documents - should read from disk
    let loadedDocs1 = try await provider.loadDocuments()
    #expect(loadedDocs1.count == 1)
    #expect(loadedDocs1[0].id == doc1.id)
    #expect(loadedDocs1[0].text == doc1.text)
    #expect(loadedDocs1[0].embedding == doc1.embedding)

    // Save another document
    let doc2 = VecturaDocument(
      id: UUID(),
      text: "Test document 2",
      embedding: [4.0, 5.0, 6.0]
    )
    try await provider.saveDocument(doc2)

    // Load again - should now have 2 documents from disk
    let loadedDocs2 = try await provider.loadDocuments()
    #expect(loadedDocs2.count == 2)

    // Delete one document
    try await provider.deleteDocument(withID: doc1.id)

    // Verify file was deleted
    #expect(!FileManager.default.fileExists(atPath: fileURL.path))

    // Load again - should only have 1 document
    let loadedDocs3 = try await provider.loadDocuments()
    #expect(loadedDocs3.count == 1)
    #expect(loadedDocs3[0].id == doc2.id)

    // Update document
    let updatedDoc2 = VecturaDocument(
      id: doc2.id,
      text: "Updated document 2",
      embedding: [7.0, 8.0, 9.0]
    )
    try await provider.updateDocument(updatedDoc2)

    // Load again - should have updated content
    let loadedDocs4 = try await provider.loadDocuments()
    #expect(loadedDocs4.count == 1)
    #expect(loadedDocs4[0].text == "Updated document 2")
    #expect(loadedDocs4[0].embedding == [7.0, 8.0, 9.0])
  }

  @Test("FileStorageProvider falls back to disk when cache is only partially populated")
  func fileStorageProviderPartialCacheFallsBackToDisk() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let seedProvider = try FileStorageProvider(storageDirectory: directory)
    let existingDoc = VecturaDocument(
      id: UUID(),
      text: "Persisted before cache warms",
      embedding: [1.0, 0.0, 0.0]
    )
    try await seedProvider.saveDocument(existingDoc)

    let provider = try FileStorageProvider(storageDirectory: directory)
    let newDoc = VecturaDocument(
      id: UUID(),
      text: "Added after restart",
      embedding: [0.0, 1.0, 0.0]
    )
    try await provider.saveDocument(newDoc)

    let found = try await provider.getDocument(id: existingDoc.id)
    #expect(found?.id == existingDoc.id)
    #expect(found?.text == existingDoc.text)
    #expect(try await provider.documentExists(id: existingDoc.id) == true)
    #expect(try await provider.getTotalDocumentCount() == 2)

    let loadedDocs = try await provider.loadDocuments()
    #expect(loadedDocs.count == 2)
    #expect(Set(loadedDocs.map(\.id)) == Set([existingDoc.id, newDoc.id]))
  }
}

/// A simple in-memory storage provider for testing custom storage implementations.
actor InMemoryStorageProvider: VecturaStorage {
  private var documents: [UUID: VecturaDocument] = [:]

  var documentCount: Int {
    documents.count
  }

  func createStorageDirectoryIfNeeded() async throws {
    // No-op for in-memory storage
  }

  func loadDocuments() async throws -> [VecturaDocument] {
    Array(documents.values)
  }

  func saveDocument(_ document: VecturaDocument) async throws {
    documents[document.id] = document
  }

  func deleteDocument(withID id: UUID) async throws {
    documents.removeValue(forKey: id)
  }

  func updateDocument(_ document: VecturaDocument) async throws {
    documents[document.id] = document
  }
}
