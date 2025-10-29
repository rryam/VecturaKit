import Foundation
import Testing
import Embeddings
@testable import VecturaKit

@Suite("VecturaKit")
struct VecturaKitTests {
    private func makeTestDirectory() throws -> (URL, () -> Void) {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appendingPathComponent("VecturaKitTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o755]
        )
        let cleanup = {
            _ = try? FileManager.default.removeItem(at: directory)
        }
        return (directory, cleanup)
    }

    private func makeVecturaConfig(
        name: String = "test-db-\(UUID().uuidString)",
        dimension: Int? = nil,
        searchOptions: VecturaConfig.SearchOptions = .init()
    ) throws -> (VecturaConfig, () -> Void) {
        let (directory, cleanup) = try makeTestDirectory()
        let config = VecturaConfig(
            name: name,
            directoryURL: directory,
            dimension: dimension,
            searchOptions: searchOptions
        )
        return (config, cleanup)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    private func makeEmbedder(modelSource: VecturaModelSource = .default) -> SwiftEmbedder {
        SwiftEmbedder(modelSource: modelSource)
    }

    @Test("Add and search document")
    func addAndSearchDocument() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let text = "This is a test document"
        let id = try await vectura.addDocument(text: text)

        let results = try await vectura.search(query: "test document")
        #expect(results.count == 1)
        #expect(results[0].id == id)
        #expect(results[0].text == text)
    }

    @Test("Add multiple documents")
    func addMultipleDocuments() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump"
        ]

        let ids = try await vectura.addDocuments(texts: documents)
        #expect(ids.count == 3)

        let results = try await vectura.search(query: "quick jumping animals")
        #expect(results.count >= 2)
        #expect(results[0].score > results[1].score)
    }

    @Test("Persistence across instances")
    func persistence() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let databaseName = UUID().uuidString
        let (config, cleanup) = try makeVecturaConfig(name: "test-db-\(databaseName)")
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let texts = ["Document 1", "Document 2"]
        let ids = try await vectura.addDocuments(texts: texts)

        let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder())
        let results = try await newVectura.search(query: "Document")
        #expect(results.count == 2)
        #expect(ids.contains(results[0].id))
        #expect(ids.contains(results[1].id))
    }

    @Test("Search threshold reduces results")
    func searchThreshold() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let documents = [
            "Very relevant document about cats",
            "Somewhat relevant about pets",
            "Completely irrelevant about weather"
        ]
        _ = try await vectura.addDocuments(texts: documents)

        let results = try await vectura.search(query: "cats and pets", threshold: 0.8)
        #expect(results.count < 3)
    }

    @Test("Custom identifiers are preserved")
    func customIds() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let customId = UUID()
        let text = "Document with custom ID"

        let resultId = try await vectura.addDocument(text: text, id: customId)
        #expect(customId == resultId)

        let results = try await vectura.search(query: text)
        #expect(results.count == 1)
        #expect(results[0].id == customId)
    }

    @Test("Model reuse remains performant")
    func modelReuse() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let start = Date()
        for i in 1...5 {
            _ = try await vectura.addDocument(text: "Test document \(i)")
        }
        let duration = Date().timeIntervalSince(start)

        #expect(duration < 10.0)
    }

    @Test("Empty search returns no results")
    func emptySearch() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let results = try await vectura.search(query: "test query")
        #expect(results.count == 0, "Search on empty database should return no results")
    }

    @Test("Dimension mismatch surfaces error")
    func dimensionMismatch() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (wrongConfig, cleanup) = try makeVecturaConfig(
            name: "wrong-dim-db-\(UUID().uuidString)",
            dimension: 128
        )
        defer { cleanup() }
        let wrongVectura = try await VecturaKit(config: wrongConfig, embedder: makeEmbedder())

        do {
            _ = try await wrongVectura.addDocument(text: "Test document")
            Issue.record("Expected dimension mismatch error")
        } catch let error as VecturaError {
            switch error {
            case .dimensionMismatch(let expected, let got):
                #expect(expected == 128)
                #expect(got > 128)
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("Duplicate identifiers overwrite documents")
    func duplicateIds() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let id = UUID()
        let text1 = "First document"
        let text2 = "Second document"

        _ = try await vectura.addDocument(text: text1, id: id)
        _ = try await vectura.addDocument(text: text2, id: id)

        let results = try await vectura.search(query: text2)
        #expect(results.count == 1)
        #expect(results[0].text == text2)
    }

    @Test("Threshold edge cases")
    func searchThresholdEdgeCases() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        _ = try await vectura.addDocuments(texts: ["Test document"])

        let highThresholdResults = try await vectura.search(query: "completely different query", threshold: 0.95)
        #expect(highThresholdResults.count <= 1)

        let allResults = try await vectura.search(query: "completely different", threshold: 0.0)
        #expect(allResults.count >= 0)
    }

    @Test("Large number of documents")
    func largeNumberOfDocuments() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let documentCount = 100
        let documents = (0..<documentCount).map { "Test document number \($0)" }

        let ids = try await vectura.addDocuments(texts: documents)
        #expect(ids.count == documentCount)

        let results = try await vectura.search(query: "document", numResults: 10)
        #expect(results.count == 10)
    }

    @Test("Persistence after reset")
    func persistenceAfterReset() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let databaseName = UUID().uuidString
        let (config, cleanup) = try makeVecturaConfig(name: "test-db-\(databaseName)")
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        let text = "Test document"
        _ = try await vectura.addDocument(text: text)

        try await vectura.reset()

        let results = try await vectura.search(query: text)
        #expect(results.count == 0)

        let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder())
        let newResults = try await newVectura.search(query: text)
        #expect(newResults.count == 0)
    }

    @Test("Folder URL model source")
    func folderURLModelSource() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

        _ = try await Model2Vec.loadModelBundle(from: VecturaModelSource.defaultModelId)

        let url = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ).appending(path: "huggingface/models/\(VecturaModelSource.defaultModelId)")

        #expect(
            FileManager.default.fileExists(atPath: url.path(percentEncoded: false)),
            "Expected downloaded model to be available locally at \(url.path())"
        )

        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump"
        ]

        let ids = try await vectura.addDocuments(texts: documents)
        #expect(ids.count == 3)

        let results = try await vectura.search(query: "quick jumping animals")
        #expect(results.count >= 2)
        #expect(results[0].score > results[1].score)
    }

    @Test("Custom storage directory")
    func customStorageDirectory() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let customDirectoryURL = URL(filePath: NSTemporaryDirectory())
            .appending(path: "VecturaKitTest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: customDirectoryURL) }

        let databaseName = "test-\(UUID().uuidString)"
        let instance = try await VecturaKit(
            config: .init(name: databaseName, directoryURL: customDirectoryURL),
            embedder: makeEmbedder()
        )
        let text = "Test document"
        let id = UUID()
        _ = try await instance.addDocument(text: text, id: id)

        let dbDirectory = customDirectoryURL.appendingPathComponent(databaseName, isDirectory: true)
        let documentPath = dbDirectory.appendingPathComponent("\(id).json").path

        #expect(
            FileManager.default.fileExists(atPath: documentPath),
            "Expected stored document at \(documentPath)"
        )

        // Verify persistence: create new instance and load from disk
        let newInstance = try await VecturaKit(
            config: .init(name: databaseName, directoryURL: customDirectoryURL),
            embedder: makeEmbedder()
        )
        #expect(newInstance.documentCount == 1, "New instance should load document from disk")

        // Verify deletion removes file from disk
        try await newInstance.deleteDocuments(ids: [id])
        #expect(
            !FileManager.default.fileExists(atPath: documentPath),
            "Document file should be deleted from disk"
        )

        // Verify third instance sees the deletion
        let thirdInstance = try await VecturaKit(
            config: .init(name: databaseName, directoryURL: customDirectoryURL),
            embedder: makeEmbedder()
        )
        #expect(thirdInstance.documentCount == 0, "Third instance should reflect deletion")
    }

    @Test("Custom storage provider")
    func customStorageProvider() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (config, cleanup) = try makeVecturaConfig()
        defer { cleanup() }

        // Create a custom in-memory storage provider
        let customStorage = InMemoryStorageProvider()
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: customStorage)

        // Add documents
        let texts = ["Custom storage document 1", "Custom storage document 2"]
        let ids = try await vectura.addDocuments(texts: texts)
        #expect(ids.count == 2)

        // Verify documents were stored in custom provider
        #expect(customStorage.documentCount == 2)

        // Search
        let results = try await vectura.search(query: "Custom storage")
        #expect(results.count == 2)

        // Delete one document
        try await vectura.deleteDocuments(ids: [ids[0]])
        #expect(customStorage.documentCount == 1)

        // Create a new instance with the same custom storage
        let newVectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: customStorage)
        let newResults = try await newVectura.search(query: "Custom")
        #expect(newResults.count == 1)
        #expect(newResults[0].id == ids[1])
    }

    @Test("FileStorageProvider is stateless and reads from disk")
    func fileStorageProviderStateless() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let (directory, cleanup) = try makeTestDirectory()
        defer { cleanup() }

        // Create a FileStorageProvider
        let provider = try FileStorageProvider(storageDirectory: directory)

        // Initially should have no documents
        let initialDocs = try await provider.loadDocuments()
        #expect(initialDocs.count == 0)

        // Create a test document and save it
        let doc1 = VecturaDocument(
            id: UUID(),
            text: "Test document 1",
            embedding: [1.0, 2.0, 3.0]
        )
        try await provider.saveDocument(doc1)

        // Verify the file was written to disk
        let fileURL = directory.appendingPathComponent("\(doc1.id).json")
        #expect(FileManager.default.fileExists(atPath: fileURL.path))

        // Load documents - should read from disk
        let loadedDocs1 = try await provider.loadDocuments()
        #expect(loadedDocs1.count == 1)
        #expect(loadedDocs1[0].id == doc1.id)
        #expect(loadedDocs1[0].text == doc1.text)
        #expect(loadedDocs1[0].embedding == doc1.embedding)

        // Save another document
        let doc2 = VecturaDocument(
            id: UUID(),
            text: "Test document 2",
            embedding: [4.0, 5.0, 6.0]
        )
        try await provider.saveDocument(doc2)

        // Load again - should now have 2 documents from disk
        let loadedDocs2 = try await provider.loadDocuments()
        #expect(loadedDocs2.count == 2)

        // Delete one document
        try await provider.deleteDocument(withID: doc1.id)

        // Verify file was deleted
        #expect(!FileManager.default.fileExists(atPath: fileURL.path))

        // Load again - should only have 1 document
        let loadedDocs3 = try await provider.loadDocuments()
        #expect(loadedDocs3.count == 1)
        #expect(loadedDocs3[0].id == doc2.id)

        // Update document
        let updatedDoc2 = VecturaDocument(
            id: doc2.id,
            text: "Updated document 2",
            embedding: [7.0, 8.0, 9.0]
        )
        try await provider.updateDocument(updatedDoc2)

        // Load again - should have updated content
        let loadedDocs4 = try await provider.loadDocuments()
        #expect(loadedDocs4.count == 1)
        #expect(loadedDocs4[0].text == "Updated document 2")
        #expect(loadedDocs4[0].embedding == [7.0, 8.0, 9.0])
    }
}

/// A simple in-memory storage provider for testing custom storage implementations.
final class InMemoryStorageProvider: VecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]

    var documentCount: Int {
        documents.count
    }

    func createStorageDirectoryIfNeeded() async throws {
        // No-op for in-memory storage
    }

    func loadDocuments() async throws -> [VecturaDocument] {
        Array(documents.values)
    }

    func saveDocument(_ document: VecturaDocument) async throws {
        documents[document.id] = document
    }

    func deleteDocument(withID id: UUID) async throws {
        documents.removeValue(forKey: id)
    }

    func updateDocument(_ document: VecturaDocument) async throws {
        documents[document.id] = document
    }
}
