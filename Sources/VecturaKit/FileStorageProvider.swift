import Foundation
import Accelerate
import os

/// A file‑based storage provider that implements VecturaStorage using JSON files.
/// This provider maintains an in‑memory cache of documents while persisting them
/// to a specified storage directory.
/// Thread-safe implementation using actor-based concurrency.
public actor FileStorageProvider: VecturaStorage {
    /// The storage directory where JSON files are stored.
    private let storageDirectory: URL

    /// In‑memory cache of documents keyed by their UUID. Protected by actor isolation.
    private var documents: [UUID: VecturaDocument] = [:]

    /// In‑memory cache of normalized embeddings for each document. Protected by actor isolation.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]
    
    /// Logger for this storage provider.
    private let logger = Logger(subsystem: "VecturaKit", category: "FileStorage")

    /// Initializes the provider with the target storage directory.
    ///
    /// - Parameter storageDirectory: The directory URL where documents will be saved and loaded.
    public init(storageDirectory: URL) {
        self.storageDirectory = storageDirectory
    }
    
    /// Factory method to create and initialize a FileStorageProvider.
    ///
    /// - Parameter storageDirectory: The directory URL where documents will be saved and loaded.
    /// - Returns: A fully initialized FileStorageProvider.
    public static func create(storageDirectory: URL) async throws -> FileStorageProvider {
        let provider = FileStorageProvider(storageDirectory: storageDirectory)
        
        // Ensure the storage directory exists
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
        
        // Load any existing documents.
        try await provider.loadDocumentsFromStorage()
        return provider
    }

    /// Ensures that the storage directory exists.
    public func createStorageDirectoryIfNeeded() async throws {
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
    }

    /// Loads documents from in‑memory cache.
    /// This function returns the documents that were loaded during initialization.
    public func loadDocuments() async throws -> [VecturaDocument] {
        return Array(documents.values)
    }

    /// Saves a document by encoding it to JSON and writing it to disk.
    /// It also updates the in‑memory caches for the document and its normalized embedding.
    public func saveDocument(_ document: VecturaDocument) async throws {
        // Update cache
        documents[document.id] = document
        
        // Encode and write document to disk
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(document)
        let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
        try data.write(to: documentURL)
        
        // Compute and store normalized embedding
        normalizedEmbeddings[document.id] = VectorMath.normalizeL2(document.embedding)
    }

    /// Deletes a document by removing it from the in‑memory caches and deleting its file.
    public func deleteDocument(withID id: UUID) async throws {
        // Remove from caches
        documents.removeValue(forKey: id)
        normalizedEmbeddings.removeValue(forKey: id)
        
        let documentURL = storageDirectory.appendingPathComponent("\(id).json")
        try FileManager.default.removeItem(at: documentURL)
    }

    /// Updates an existing document.
    /// This is implemented by saving the updated document, which overwrites the existing file.
    public func updateDocument(_ document: VecturaDocument) async throws {
        try await saveDocument(document)
    }

    // MARK: - Private Helper Methods

    /// Loads all JSON‑encoded documents from disk into memory.
    private func loadDocumentsFromStorage() async throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil)
        let decoder = JSONDecoder()
        
        for fileURL in fileURLs where fileURL.pathExtension.lowercased() == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let doc = try decoder.decode(VecturaDocument.self, from: data)
                documents[doc.id] = doc
                
                // Compute normalized embedding and store it.
                normalizedEmbeddings[doc.id] = VectorMath.normalizeL2(doc.embedding)
            } catch {
                // Log the error with proper privacy considerations
                logger.error("Failed to load document: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

}
