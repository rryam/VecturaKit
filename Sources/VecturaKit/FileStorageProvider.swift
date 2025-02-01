import Foundation
import Accelerate

/// A file‑based storage provider that implements VecturaStorage using JSON files.
/// This provider maintains an in‑memory cache of documents while persisting them
/// to a specified storage directory.
public class FileStorageProvider: VecturaStorage {
    /// The storage directory where JSON files are stored.
    private let storageDirectory: URL

    /// In‑memory cache of documents keyed by their UUID.
    private var documents: [UUID: VecturaDocument] = [:]

    /// In‑memory cache of normalized embeddings for each document.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]

    /// Initializes the provider with the target storage directory.
    ///
    /// - Parameter storageDirectory: The directory URL where documents will be saved and loaded.
    public init(storageDirectory: URL) throws {
        self.storageDirectory = storageDirectory
        
        // Ensure the storage directory exists
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
        
        // Load any existing documents.
        try loadDocumentsFromStorage()
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
        let norm = l2Norm(document.embedding)
        var divisor = norm + 1e-9
        var normalized = [Float](repeating: 0, count: document.embedding.count)
        vDSP_vsdiv(document.embedding, 1, &divisor, &normalized, 1, vDSP_Length(document.embedding.count))
        normalizedEmbeddings[document.id] = normalized
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
    private func loadDocumentsFromStorage() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil)
        let decoder = JSONDecoder()
        
        for fileURL in fileURLs where fileURL.pathExtension.lowercased() == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let doc = try decoder.decode(VecturaDocument.self, from: data)
                documents[doc.id] = doc
                
                // Compute normalized embedding and store it.
                let norm = l2Norm(doc.embedding)
                var divisor = norm + 1e-9
                var normalized = [Float](repeating: 0, count: doc.embedding.count)
                vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
                normalizedEmbeddings[doc.id] = normalized
            } catch {
                // Log the error if needed
                print("Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }
    }

    /// Computes the L2 norm of a vector.
    private func l2Norm(_ vector: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        return sqrt(sumSquares)
    }
}
