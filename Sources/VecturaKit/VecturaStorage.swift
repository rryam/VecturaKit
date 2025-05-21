import Foundation

/// VecturaStorage protocol abstracts the persistence layer for VecturaDocuments.
///
/// It allows for multiple underlying storage implementations (e.g., File-based or SQLite)
/// without changing the higher-level API used in VecturaKit.
public protocol VecturaStorage {
    /// Prepares or creates the storage location for documents if needed.
    func createStorageDirectoryIfNeeded() async throws
    
    /// Loads the persisted documents.
    /// It should iterate through files in storage, decode them, and return documents.
    /// If individual files fail to load/decode, it should collect these errors.
    ///
    /// - Returns: A tuple containing an array of successfully loaded documents and an array of errors.
    func loadDocuments() async -> (documents: [VecturaDocument], errors: [Error])
    
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
    // func updateDocument(_ document: VecturaDocument) async throws 
    // Removing from protocol as FileStorageProvider will no longer implement it,
    // and VecturaKit will handle the delete & save sequence.
}
