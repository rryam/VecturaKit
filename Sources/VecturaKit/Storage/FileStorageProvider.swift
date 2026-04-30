import Foundation
import OSLog

/// A file‑based storage provider that implements VecturaStorage using JSON files.
/// This provider adopts CachableVecturaStorage and implements caching logic internally
/// for thread-safe document management.
public actor FileStorageProvider {

  /// The storage directory where JSON files are stored.
  private let storageDirectory: URL

  /// Logger for error reporting
  private static let logger = Logger(
    subsystem: "com.vecturakit",
    category: "FileStorageProvider"
  )

  /// Internal cache for documents (actor-isolated for thread safety).
  ///
  /// The cache may be only partially populated until the provider performs a
  /// full load from disk, so cache misses cannot be treated as authoritative
  /// unless `cacheIsFullyLoaded` is true.
  private var cache: [UUID: VecturaDocument] = [:]

  /// Whether the in-memory cache currently represents the complete dataset.
  private var cacheIsFullyLoaded = false

  /// Whether caching is enabled
  private let cacheEnabled: Bool

  /// Ensures the storage directory exists, creating it if needed.
  nonisolated private func ensureStorageDirectoryExists() throws {
    if !FileManager.default.fileExists(atPath: storageDirectory.path) {
      try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
    }
  }

  /// Initializes the provider with the target storage directory.
  ///
  /// - Parameters:
  ///   - storageDirectory: The directory URL where documents will be saved and loaded.
  ///   - cacheEnabled: Whether to enable in-memory caching (default: true)
  public init(storageDirectory: URL, cacheEnabled: Bool = true) throws {
    self.storageDirectory = storageDirectory
    self.cacheEnabled = cacheEnabled

    // Ensure the storage directory exists
    try ensureStorageDirectoryExists()
  }
}

// MARK: - VecturaStorage

extension FileStorageProvider: VecturaStorage {
  // MARK: - VecturaStorage Protocol

  /// Ensures that the storage directory exists.
  public func createStorageDirectoryIfNeeded() async throws {
    // Use nonisolated helper since FileManager operations are thread-safe
    try ensureStorageDirectoryExists()
  }

  /// Loads all documents, using cache if available
  public func loadDocuments() async throws -> [VecturaDocument] {
    if cacheEnabled && cacheIsFullyLoaded {
      return Array(cache.values)
    }

    let documents = try await loadDocumentsFromStorage()

    if cacheEnabled {
      replaceCache(with: documents, fullyLoaded: true)
    }

    return documents
  }

  /// Returns total document count without decoding all document files.
  public func getTotalDocumentCount() async throws -> Int {
    // Count directly from disk because overlapping loads and writes can temporarily leave
    // the cache in a stale-but-fully-loaded state.
    let fileURLs = try FileManager.default.contentsOfDirectory(
      at: storageDirectory,
      includingPropertiesForKeys: nil
    )
    return fileURLs.reduce(into: 0) { count, fileURL in
      if fileURL.pathExtension.lowercased() == "json" {
        count += 1
      }
    }
  }

  /// Saves a document and updates cache
  public func saveDocument(_ document: VecturaDocument) async throws {
    try await saveDocumentToStorage(document)

    if cacheEnabled {
      cache[document.id] = document
    }
  }

  /// Deletes a document and removes from cache
  public func deleteDocument(withID id: UUID) async throws {
    try await deleteDocumentFromStorage(withID: id)

    if cacheEnabled {
      cache.removeValue(forKey: id)
    }
  }

  /// Updates a document and refreshes cache
  public func updateDocument(_ document: VecturaDocument) async throws {
    try await saveDocumentToStorage(document)

    if cacheEnabled {
      cache[document.id] = document
    }
  }

  /// Saves multiple documents with throttled concurrent file writes.
  ///
  /// Uses nonisolated file I/O to achieve true parallelism, then updates
  /// the cache after all writes complete.
  public func saveDocuments(_ documents: [VecturaDocument]) async throws {
    guard !documents.isEmpty else {
      return
    }

    let directory = storageDirectory

    struct SaveOutcome: Sendable {
      let document: VecturaDocument
      let errorDescription: String?
    }

    // Perform concurrent file writes outside actor isolation
    let outcomes = await documents.concurrentMap(maxConcurrency: Self.maxConcurrentFileOperations) { document in
      do {
        try Self.writeDocumentToFile(document, in: directory)
        return SaveOutcome(document: document, errorDescription: nil)
      } catch {
        return SaveOutcome(document: document, errorDescription: error.localizedDescription)
      }
    }

    let failures = outcomes.filter { $0.errorDescription != nil }
    if !failures.isEmpty {
      for failure in failures {
        Self.logger.warning(
          "Failed to save document \(failure.document.id): \(failure.errorDescription ?? "Unknown error")"
        )
      }
    }

    if cacheEnabled {
      if failures.isEmpty {
        for outcome in outcomes {
          cache[outcome.document.id] = outcome.document
        }
      } else {
        do {
          let refreshed = try await loadDocumentsFromStorage()
          replaceCache(with: refreshed, fullyLoaded: true)
        } catch {
          cache.removeAll()
          cacheIsFullyLoaded = false
        }
      }
    }

    if let firstFailure = failures.first {
      throw VecturaError.loadFailed(
        "Failed to save \(failures.count) document(s). First error: \(firstFailure.errorDescription ?? "Unknown error")"
      )
    }
  }

  /// Writes a document to disk without actor isolation.
  ///
  /// This allows concurrent file I/O when called from multiple tasks.
  nonisolated private static func writeDocumentToFile(
    _ document: VecturaDocument,
    in directory: URL
  ) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    let data = try encoder.encode(document)
    let documentURL = directory.appendingPathComponent("\(document.id).json")

    #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
    try data.write(to: documentURL, options: [.atomic, .completeFileProtection])
    #else
    try data.write(to: documentURL, options: .atomic)
    #endif

    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: documentURL.path(percentEncoded: false)
    )
  }
}

// MARK: - CachableVecturaStorage

extension FileStorageProvider: CachableVecturaStorage {

  /// Maximum number of concurrent file operations to prevent resource exhaustion.
  private static let maxConcurrentFileOperations = 50

  /// Loads all documents from disk by reading JSON files (bypasses cache).
  ///
  /// Uses throttled concurrency to prevent file descriptor exhaustion
  /// when loading large numbers of documents.
  public func loadDocumentsFromStorage() async throws -> [VecturaDocument] {
    let fileURLs = try FileManager.default.contentsOfDirectory(
      at: storageDirectory,
      includingPropertiesForKeys: nil
    )

    let jsonFileURLs = fileURLs.filter { $0.pathExtension.lowercased() == "json" }

    struct LoadOutcome: Sendable {
      let document: VecturaDocument?
      let path: String
      let errorDescription: String?
    }

    let outcomes = await jsonFileURLs.concurrentMap(maxConcurrency: Self.maxConcurrentFileOperations) { fileURL in
      let path = fileURL.path(percentEncoded: false)
      do {
        let data = try Data(contentsOf: fileURL)
        let document = try JSONDecoder().decode(VecturaDocument.self, from: data)
        return LoadOutcome(document: document, path: path, errorDescription: nil)
      } catch {
        return LoadOutcome(document: nil, path: path, errorDescription: error.localizedDescription)
      }
    }

    let failures = outcomes.filter { $0.document == nil }
    if !failures.isEmpty {
      for failure in failures {
        Self.logger.warning(
          "Failed to load document at \(failure.path): \(failure.errorDescription ?? "Unknown error")"
        )
      }

      let firstFailure = failures[0]
      throw VecturaError.loadFailed(
        "Failed to load \(failures.count) document(s). First error at \(firstFailure.path): "
          + "\(firstFailure.errorDescription ?? "Unknown error")"
      )
    }

    return outcomes.compactMap(\.document)
  }

  /// Saves a document by encoding it to JSON and writing it to disk (bypasses cache).
  public func saveDocumentToStorage(_ document: VecturaDocument) async throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    let data = try encoder.encode(document)
    let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")

    // Write with secure file protection on supported platforms (iOS-family only)
    #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
    try data.write(to: documentURL, options: [.atomic, .completeFileProtection])
    #else
    try data.write(to: documentURL, options: .atomic)
    #endif

    // Set restrictive file permissions (owner read/write only)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: documentURL.path(percentEncoded: false)
    )

    // Verify permissions were set correctly on macOS (iOS/tvOS have different security model)
    #if !os(iOS) && !os(tvOS) && !os(watchOS) && !os(visionOS)
    let attributes = try FileManager.default.attributesOfItem(atPath: documentURL.path(percentEncoded: false))
    if let permissions = attributes[.posixPermissions] as? NSNumber {
      if permissions.uint16Value != 0o600 {
        Self.logger.warning("File permissions verification failed for \(documentURL.path(percentEncoded: false))")
      }
    }
    #endif
  }

  /// Deletes a document by removing its file from disk (bypasses cache).
  ///
  /// If the file does not exist, this is treated as a no-op rather than an error.
  /// This makes `deleteDocuments(ids:)` idempotent and prevents partial-delete
  /// failures when an ID is not present in storage.
  public func deleteDocumentFromStorage(withID id: UUID) async throws {
    let documentURL = storageDirectory.appendingPathComponent("\(id).json")
    guard FileManager.default.fileExists(atPath: documentURL.path(percentEncoded: false)) else {
      return
    }
    try FileManager.default.removeItem(at: documentURL)
  }
}

// MARK: - Efficient Single-Document Lookup

extension FileStorageProvider {

  /// Returns a single document by ID without loading all documents from disk.
  ///
  /// Lookup order:
  /// 1. In-memory cache (O(1), no I/O) when the document is already cached
  /// 2. Single targeted file read for cache misses or when the cache is cold
  public func getDocument(id: UUID) async throws -> VecturaDocument? {
    if cacheEnabled {
      if let cachedDocument = cache[id] {
        return cachedDocument
      }
      if cacheIsFullyLoaded {
        return nil
      }
    }
    let fileURL = storageDirectory.appendingPathComponent("\(id).json")
    guard FileManager.default.fileExists(atPath: fileURL.path(percentEncoded: false)) else {
      return nil
    }
    let data = try Data(contentsOf: fileURL)
    return try JSONDecoder().decode(VecturaDocument.self, from: data)
  }

  /// Returns whether a document with the given ID exists, without decoding its contents.
  ///
  /// Lookup order:
  /// 1. In-memory cache (O(1), no I/O) when the document is already cached
  /// 2. File-existence check (O(1), no JSON decoding) for cache misses or when the cache is cold
  public func documentExists(id: UUID) async throws -> Bool {
    if cacheEnabled {
      if cache[id] != nil {
        return true
      }
      if cacheIsFullyLoaded {
        return false
      }
    }
    let fileURL = storageDirectory.appendingPathComponent("\(id).json")
    return FileManager.default.fileExists(atPath: fileURL.path(percentEncoded: false))
  }

  private func replaceCache(with documents: [VecturaDocument], fullyLoaded: Bool) {
    cache = documents.reduce(into: [:]) { dict, doc in
      if dict[doc.id] != nil {
        Self.logger.warning("Duplicate document ID found during cache load: \(doc.id)")
      }
      dict[doc.id] = doc
    }
    cacheIsFullyLoaded = fullyLoaded
  }
}
