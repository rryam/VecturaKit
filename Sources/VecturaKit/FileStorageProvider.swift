import Foundation
import Accelerate

/// A file‑based storage provider that implements VecturaStorage using JSON files.
/// This provider maintains an in‑memory cache of documents while persisting them
/// to a specified storage directory.
public class FileStorageProvider: VecturaStorage {
    /// The storage directory where JSON files are stored.
    private let storageDirectory: URL

    /// Initializes the provider with the target storage directory.
    ///
    /// - Parameter storageDirectory: The directory URL where documents will be saved and loaded.
    public init(storageDirectory: URL) throws {
        self.storageDirectory = storageDirectory
        
        // Ensure the storage directory exists
        if !FileManager.default.fileExists(atPath: storageDirectory.path(percentEncoded: false)) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
    }

    /// Ensures that the storage directory exists.
    public func createStorageDirectoryIfNeeded() async throws {
        if !FileManager.default.fileExists(atPath: storageDirectory.path(percentEncoded: false)) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
    }

    /// Loads all JSON-encoded documents from disk.
    /// This method iterates through files in the storage directory, attempts to decode them into
    /// `VecturaDocument` objects, and collects any errors encountered during the process.
    ///
    /// - Returns: A tuple containing an array of successfully loaded documents and an array of errors.
    public func loadDocuments() async -> (documents: [VecturaDocument], errors: [Error]) {
        var loadedDocuments: [VecturaDocument] = []
        var loadingErrors: [Error] = []
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil)
            let decoder = JSONDecoder()
            
            for fileURL in fileURLs where fileURL.pathExtension.lowercased() == "json" {
                do {
                    let data = try Data(contentsOf: fileURL)
                    let doc = try decoder.decode(VecturaDocument.self, from: data)
                    loadedDocuments.append(doc)
                } catch {
                    // Collect error for this specific file
                    loadingErrors.append(VecturaError.loadFailed("Failed to load or decode \(fileURL.lastPathComponent): \(error.localizedDescription)"))
                }
            }
        } catch {
            // Error reading directory contents
            loadingErrors.append(VecturaError.loadFailed("Failed to read contents of storage directory: \(error.localizedDescription)"))
        }
        
        return (loadedDocuments, loadingErrors)
    }

    /// Saves a document by encoding it to JSON and writing it to disk.
    public func saveDocument(_ document: VecturaDocument) async throws {
        // Encode and write document to disk
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(document)
        let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
        try data.write(to: documentURL)
    }

    /// Deletes a document file from disk.
    public func deleteDocument(withID id: UUID) async throws {
        let documentURL = storageDirectory.appendingPathComponent("\(id).json")
        // Check if file exists before attempting to delete to avoid error if already deleted or never existed
        if FileManager.default.fileExists(atPath: documentURL.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: documentURL)
        }
    }

    // updateDocument method is removed as per instructions.
    // l2Norm method is removed as per instructions.
}
