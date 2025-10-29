import Foundation

/// A file‑based storage provider that implements VecturaStorage using JSON files.
/// This provider persists documents to disk without maintaining an in-memory cache,
/// as caching is handled by the VecturaKit layer.
public final class FileStorageProvider: VecturaStorage {
    /// The storage directory where JSON files are stored.
    private let storageDirectory: URL

    /// Initializes the provider with the target storage directory.
    ///
    /// - Parameter storageDirectory: The directory URL where documents will be saved and loaded.
    public init(storageDirectory: URL) throws {
        self.storageDirectory = storageDirectory

        // Ensure the storage directory exists
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
    }

    /// Ensures that the storage directory exists.
    public func createStorageDirectoryIfNeeded() async throws {
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        }
    }

    /// Loads all documents from disk by reading JSON files.
    public func loadDocuments() async throws -> [VecturaDocument] {
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
                        print("Failed to load document at \(fileURL.path): \(error.localizedDescription)")
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

    /// Saves a document by encoding it to JSON and writing it to disk.
    public func saveDocument(_ document: VecturaDocument) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(document)
        let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
        try data.write(to: documentURL)
    }

    /// Deletes a document by removing its file from disk.
    public func deleteDocument(withID id: UUID) async throws {
        let documentURL = storageDirectory.appendingPathComponent("\(id).json")
        try FileManager.default.removeItem(at: documentURL)
    }

    /// Updates an existing document.
    /// This is implemented by saving the updated document, which overwrites the existing file.
    public func updateDocument(_ document: VecturaDocument) async throws {
        try await saveDocument(document)
    }
}
