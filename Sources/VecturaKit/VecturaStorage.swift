import Foundation

/// VecturaStorage protocol abstracts the persistence layer for VecturaDocuments.
///
/// It allows for multiple underlying storage implementations (e.g., File-based or SQLite)
/// without changing the higher-level API used in VecturaKit.
public protocol VecturaStorage {
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
}
