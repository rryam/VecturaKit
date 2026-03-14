import Foundation

/// VecturaStorage protocol abstracts the persistence layer for VecturaDocuments.
///
/// It allows for multiple underlying storage implementations (e.g., File-based or SQLite)
/// without changing the higher-level API used in VecturaKit.
public protocol VecturaStorage: Sendable {

  /// Prepares or creates the storage location for documents if needed.
  func createStorageDirectoryIfNeeded() async throws

  /// Loads the persisted documents.
  ///
  /// - Returns: An array of VecturaDocument.
  func loadDocuments() async throws -> [VecturaDocument]

  /// Saves a document.
  ///
  /// - Parameter document: The document to save.
  func saveDocument(_ document: VecturaDocument) async throws

  /// Deletes a document by its unique identifier.
  ///
  /// - Parameter id: The identifier of the document to be deleted.
  func deleteDocument(withID id: UUID) async throws

  /// Updates an existing document. The document is replaced or modified as needed.
  ///
  /// - Parameter document: The updated document.
  func updateDocument(_ document: VecturaDocument) async throws

  /// Returns the total number of documents in storage.
  ///
  /// - Returns: The total document count.
  func getTotalDocumentCount() async throws -> Int

  /// Saves multiple documents in batch.
  ///
  /// Storage providers can override this for optimized batch operations.
  /// The default implementation calls saveDocument sequentially.
  ///
  /// - Parameter documents: The documents to save.
  func saveDocuments(_ documents: [VecturaDocument]) async throws

  /// Returns a single document by its ID, or nil if not found.
  ///
  /// Storage providers can override this for efficient single-document lookup.
  /// The default implementation loads all documents and filters by ID.
  ///
  /// - Parameter id: The unique identifier of the document to retrieve.
  /// - Returns: The document if found, nil otherwise.
  func getDocument(id: UUID) async throws -> VecturaDocument?

  /// Returns whether a document with the given ID exists in storage.
  ///
  /// Storage providers can override this for efficient existence checks.
  /// The default implementation delegates to `getDocument(id:)`.
  ///
  /// - Parameter id: The unique identifier to check.
  /// - Returns: True if the document exists, false otherwise.
  func documentExists(id: UUID) async throws -> Bool
}

// MARK: - Default Implementation

extension VecturaStorage {
  /// Default implementation that loads all documents and counts them.
  ///
  /// Storage implementations can override this for better performance
  /// (e.g., SQL COUNT(*) query instead of loading all documents).
  public func getTotalDocumentCount() async throws -> Int {
    return try await loadDocuments().count
  }

  /// Default implementation that saves documents sequentially.
  ///
  /// Storage implementations can override this for concurrent I/O.
  public func saveDocuments(_ documents: [VecturaDocument]) async throws {
    for document in documents {
      try await saveDocument(document)
    }
  }

  /// Default implementation that loads all documents and finds by ID.
  ///
  /// Storage implementations should override this for better performance
  /// (e.g., a cache lookup or a targeted file read instead of loading everything).
  public func getDocument(id: UUID) async throws -> VecturaDocument? {
    let docs = try await loadDocuments()
    return docs.first { $0.id == id }
  }

  /// Default implementation that delegates to `getDocument(id:)`.
  ///
  /// Storage implementations can override this for a cheaper check that
  /// avoids decoding the document (e.g., a file-existence check).
  public func documentExists(id: UUID) async throws -> Bool {
    return try await getDocument(id: id) != nil
  }
}
