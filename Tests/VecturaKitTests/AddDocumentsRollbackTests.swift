import Foundation
import Testing
@testable import VecturaKit

/// Regression tests for the failure paths of `addDocuments`.
@Suite("addDocuments Rollback")
struct AddDocumentsRollbackTests {

  private struct FixedEmbedder: VecturaEmbedder {
    var dimension: Int {
      get async throws { 3 }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
      Array(repeating: [1.0, 0.0, 0.0], count: texts.count)
    }
  }

  /// Storage whose batch save writes some documents and then fails, mirroring
  /// a real partial failure (disk full, permissions) inside `saveDocuments`.
  private actor PartiallyFailingStorage: VecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]
    private let failAfterCount: Int
    private var shouldFailSaves = true

    init(failAfterCount: Int) {
      self.failAfterCount = failAfterCount
    }

    var storedDocuments: [UUID: VecturaDocument] {
      documents
    }

    func stopFailing() {
      shouldFailSaves = false
    }

    func startFailing() {
      shouldFailSaves = true
    }

    func createStorageDirectoryIfNeeded() async throws {}

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

    func saveDocuments(_ documents: [VecturaDocument]) async throws {
      guard shouldFailSaves else {
        for document in documents {
          self.documents[document.id] = document
        }
        return
      }

      // Commit the first N writes, then fail — exactly what a partial batch
      // failure leaves behind on disk.
      for document in documents.prefix(failAfterCount) {
        self.documents[document.id] = document
      }
      throw VecturaError.loadFailed("simulated partial save failure")
    }
  }

  /// A partial batch save must not destroy the documents it overwrote.
  ///
  /// `saveDocuments` used to run outside the rollback scope, so a save that
  /// failed midway left the already-written documents in place — including
  /// ones that had overwritten existing content — while reporting failure.
  @Test("Partial save failure restores overwritten documents")
  func partialSaveFailureRestoresOverwrittenDocuments() async throws {
    let storage = PartiallyFailingStorage(failAfterCount: 1)
    let config = try VecturaConfig(name: "rollback-overwrite", dimension: 3)

    // Seed an existing document through a successful save.
    await storage.stopFailing()
    let seeded = try await VecturaKit(config: config, embedder: FixedEmbedder(), storageProvider: storage)
    let existingID = try await seeded.addDocument(text: "original text")

    // Now the next batch save writes the first document and then fails.
    await storage.startFailing()
    let vectura = try await VecturaKit(config: config, embedder: FixedEmbedder(), storageProvider: storage)

    await #expect(throws: (any Error).self) {
      _ = try await vectura.addDocuments(
        texts: ["overwriting text", "second text"],
        ids: [existingID, UUID()]
      )
    }

    // The overwritten document must be back to its original content.
    let stored = await storage.storedDocuments
    #expect(stored[existingID]?.text == "original text")
    #expect(stored.count == 1)
  }

  /// A partial batch save of brand-new documents must not leave orphans that
  /// the caller was told were never added.
  @Test("Partial save failure removes orphaned new documents")
  func partialSaveFailureRemovesOrphans() async throws {
    let storage = PartiallyFailingStorage(failAfterCount: 1)
    let config = try VecturaConfig(name: "rollback-orphan", dimension: 3)
    let vectura = try await VecturaKit(config: config, embedder: FixedEmbedder(), storageProvider: storage)

    await #expect(throws: (any Error).self) {
      _ = try await vectura.addDocuments(texts: ["first", "second"])
    }

    let stored = await storage.storedDocuments
    #expect(stored.isEmpty)
  }

  /// Duplicate IDs in one batch collapse into a single document, so the call
  /// would otherwise report more documents added than actually exist.
  @Test("Duplicate IDs within a batch are rejected")
  func duplicateIDsWithinBatchAreRejected() async throws {
    let storage = PartiallyFailingStorage(failAfterCount: 0)
    await storage.stopFailing()
    let config = try VecturaConfig(name: "duplicate-ids", dimension: 3)
    let vectura = try await VecturaKit(config: config, embedder: FixedEmbedder(), storageProvider: storage)

    let sharedID = UUID()
    await #expect(throws: VecturaError.self) {
      _ = try await vectura.addDocuments(
        texts: ["first", "second"],
        ids: [sharedID, sharedID]
      )
    }

    let stored = await storage.storedDocuments
    #expect(stored.isEmpty)
  }
}

// MARK: - Concurrency helpers

@Suite("Concurrent Collection Helpers")
struct ConcurrentCollectionHelperTests {

  /// A non-positive window used to seed no tasks at all, silently returning an
  /// empty result rather than processing the elements.
  @Test("concurrentMap with non-positive concurrency still processes elements")
  func concurrentMapWithNonPositiveConcurrencyProcessesElements() async {
    let doubled: [Int] = await [1, 2, 3].concurrentMap(maxConcurrency: 0) { value in
      value * 2
    }

    #expect(doubled.sorted() == [2, 4, 6])
  }

  @Test("concurrentForEach with non-positive concurrency still runs side effects")
  func concurrentForEachWithNonPositiveConcurrencyRunsSideEffects() async throws {
    let counter = Counter()

    try await [1, 2, 3].concurrentForEach(maxConcurrency: -1) { _ in
      await counter.increment()
    }

    #expect(await counter.value == 3)
  }

  @Test("orderedConcurrentMap preserves input order")
  func orderedConcurrentMapPreservesOrder() async throws {
    let results = try await [1, 2, 3, 4, 5].orderedConcurrentMap(maxConcurrency: 2) { value in
      value * 10
    }

    #expect(results == [10, 20, 30, 40, 50])
  }

  private actor Counter {
    private(set) var value = 0

    func increment() {
      value += 1
    }
  }
}
