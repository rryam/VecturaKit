import Foundation
import Testing
@testable import VecturaKit

/// Regression tests for index/cache coherence when a mutation interleaves with
/// an in-flight load from storage.
@Suite("Index Coherence")
struct IndexCoherenceTests {

  /// Storage whose `loadDocuments()` blocks until the test opens a gate, so a
  /// mutation can be injected while the load is suspended.
  private actor GatedStorage: VecturaStorage {
    private var documents: [VecturaDocument]
    private var gateOpened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(documents: [VecturaDocument]) {
      self.documents = documents
    }

    /// Number of loads currently suspended on the gate.
    var waiterCount: Int {
      waiters.count
    }

    func append(_ document: VecturaDocument) {
      documents.append(document)
    }

    func openGate() {
      gateOpened = true
      let pending = waiters
      waiters.removeAll()
      for waiter in pending {
        waiter.resume()
      }
    }

    /// Releases one suspended load, so tests can control the resume order.
    func releaseWaiter(at position: Int) {
      guard waiters.indices.contains(position) else {
        return
      }
      waiters.remove(at: position).resume()
    }

    func createStorageDirectoryIfNeeded() async throws {}

    func loadDocuments() async throws -> [VecturaDocument] {
      // Snapshot before suspending, mirroring a real read that misses writes
      // landing after the read began.
      let snapshot = documents

      if !gateOpened {
        await withCheckedContinuation { continuation in
          waiters.append(continuation)
        }
      }

      return snapshot
    }

    func saveDocument(_ document: VecturaDocument) async throws {}
    func deleteDocument(withID id: UUID) async throws {}
    func updateDocument(_ document: VecturaDocument) async throws {}
  }

  /// A document indexed while a rebuild is in flight must not be silently lost.
  ///
  /// `BM25SearchEngine.search` used to clear `needsRebuild` *after* awaiting
  /// storage, discarding the dirty mark set by an `indexDocument` that
  /// interleaved during that await. The document was then absent from the index
  /// permanently — no later search would ever rebuild and pick it up.
  @Test("Document indexed during an in-flight rebuild is not lost")
  func documentIndexedDuringRebuildIsNotLost() async throws {
    let existing = VecturaDocument(text: "apple orchard", embedding: [1, 0])
    let storage = GatedStorage(documents: [existing])
    let engine = BM25SearchEngine()

    // Mark the index dirty without building it, as the first add does.
    try await engine.indexDocument(existing)

    // Start a search: it enters the rebuild path and suspends inside loadDocuments().
    let searchTask = Task {
      try await engine.search(
        query: .text("apple"),
        storage: storage,
        options: try SearchOptions(numResults: 10)
      )
    }

    // Wait until the load is actually in flight and suspended on the gate.
    while await storage.waiterCount == 0 {
      await Task.yield()
    }

    // A concurrent add lands in storage and marks the index dirty while the
    // rebuild's snapshot (taken above) is already fixed and does not contain it.
    let added = VecturaDocument(text: "banana grove", embedding: [0, 1])
    await storage.append(added)
    try await engine.indexDocument(added)

    await storage.openGate()
    _ = try await searchTask.value

    // The dirty mark must have survived, so this search rebuilds and finds it.
    let results = try await engine.search(
      query: .text("banana"),
      storage: storage,
      options: try SearchOptions(numResults: 10)
    )

    #expect(results.map(\.id) == [added.id])
  }

  /// Two rebuilds can be in flight at once and resume in either order. The one
  /// that loaded the older snapshot must not install it over the newer index,
  /// or the documents added in between vanish until the index is unloaded.
  @Test("Slower concurrent rebuild does not clobber a newer index")
  func slowerConcurrentRebuildDoesNotClobberNewerIndex() async throws {
    let existing = VecturaDocument(text: "apple orchard", embedding: [1, 0])
    let storage = GatedStorage(documents: [existing])
    let engine = BM25SearchEngine()

    try await engine.indexDocument(existing)

    // First rebuild suspends holding a snapshot without the document below.
    let firstSearch = Task {
      try await engine.search(
        query: .text("apple"),
        storage: storage,
        options: try SearchOptions(numResults: 10)
      )
    }
    while await storage.waiterCount < 1 {
      await Task.yield()
    }

    let added = VecturaDocument(text: "banana grove", embedding: [0, 1])
    await storage.append(added)
    try await engine.indexDocument(added)

    // Second rebuild suspends holding a snapshot that does contain it.
    let secondSearch = Task {
      try await engine.search(
        query: .text("banana"),
        storage: storage,
        options: try SearchOptions(numResults: 10)
      )
    }
    while await storage.waiterCount < 2 {
      await Task.yield()
    }

    // Resume the newer rebuild first, then the older one.
    await storage.releaseWaiter(at: 1)
    _ = try await secondSearch.value
    await storage.releaseWaiter(at: 0)
    _ = try await firstSearch.value

    // The older snapshot must have been discarded rather than installed over
    // the newer index, which still holds both documents.
    #expect(await engine.indexedDocumentCount == 2)

    await storage.openGate()

    let results = try await engine.search(
      query: .text("banana"),
      storage: storage,
      options: try SearchOptions(numResults: 10)
    )

    #expect(results.map(\.id) == [added.id])
  }

  /// A failed rebuild must leave the index dirty so the next search retries.
  @Test("Failed rebuild leaves the index dirty for the next search")
  func failedRebuildLeavesIndexDirty() async throws {
    let doc = VecturaDocument(text: "resilient document", embedding: [1, 0])
    let storage = FlakyStorage(documents: [doc])
    let engine = BM25SearchEngine()

    try await engine.indexDocument(doc)

    await #expect(throws: (any Error).self) {
      _ = try await engine.search(
        query: .text("resilient"),
        storage: storage,
        options: try SearchOptions(numResults: 10)
      )
    }

    // Second search must retry the rebuild rather than serve an empty index.
    let results = try await engine.search(
      query: .text("resilient"),
      storage: storage,
      options: try SearchOptions(numResults: 10)
    )

    #expect(results.map(\.id) == [doc.id])
  }

  /// Storage that fails the first load and succeeds afterwards.
  private actor FlakyStorage: VecturaStorage {
    private let documents: [VecturaDocument]
    private var hasFailedOnce = false

    init(documents: [VecturaDocument]) {
      self.documents = documents
    }

    func createStorageDirectoryIfNeeded() async throws {}

    func loadDocuments() async throws -> [VecturaDocument] {
      if !hasFailedOnce {
        hasFailedOnce = true
        throw VecturaError.loadFailed("transient failure")
      }
      return documents
    }

    func saveDocument(_ document: VecturaDocument) async throws {}
    func deleteDocument(withID id: UUID) async throws {}
    func updateDocument(_ document: VecturaDocument) async throws {}
  }
}
