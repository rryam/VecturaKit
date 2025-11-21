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

  /// Internal cache for documents (actor-isolated for thread safety)
  private var cache: [UUID: VecturaDocument] = [:]

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
    if cacheEnabled && !cache.isEmpty {
      return Array(cache.values)
    }

    let documents = try await loadDocumentsFromStorage()

    if cacheEnabled {
      cache = documents.reduce(into: [:]) { dict, doc in
        if dict[doc.id] != nil {
          Self.logger.warning("Duplicate document ID found during cache load: \(doc.id)")
        }
        dict[doc.id] = doc
      }
    }

    return documents
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
}

// MARK: - CachableVecturaStorage

extension FileStorageProvider: CachableVecturaStorage {
  /// Loads all documents from disk by reading JSON files (bypasses cache).
  public func loadDocumentsFromStorage() async throws -> [VecturaDocument] {
    let fileURLs = try FileManager.default.contentsOfDirectory(
      at: storageDirectory,
      includingPropertiesForKeys: nil
    )

    let jsonFileURLs = fileURLs.filter { $0.pathExtension.lowercased() == "json" }

    return await withTaskGroup(of: VecturaDocument?.self, returning: [VecturaDocument].self) { group in
      for fileURL in jsonFileURLs {
        group.addTask {
          do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            return try decoder.decode(VecturaDocument.self, from: data)
          } catch {
            let path = fileURL.path(percentEncoded: false)
            Self.logger.warning(
              "Failed to load document at \(path): \(error.localizedDescription)"
            )
            return nil
          }
        }
      }

      var documents: [VecturaDocument] = []
      documents.reserveCapacity(jsonFileURLs.count)
      for await document in group {
        if let document {
          documents.append(document)
        }
      }
      return documents
    }
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
  public func deleteDocumentFromStorage(withID id: UUID) async throws {
    let documentURL = storageDirectory.appendingPathComponent("\(id).json")
    try FileManager.default.removeItem(at: documentURL)
  }
}
