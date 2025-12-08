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
}
