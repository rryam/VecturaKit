import Foundation
import Testing
import Embeddings
@testable import VecturaKit

@Suite("VecturaKit")
struct VecturaKitTests {
    @Test("Add and search document")
    func addAndSearchDocument() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let text = "This is a test document"
        let id = try await vectura.addDocument(text: text)

        let results = try await vectura.search(query: "test document")
        #expect(results.count == 1)
        #expect(results[0].id == id)
        #expect(results[0].text == text)

        try await vectura.reset()
    }

    @Test("Add multiple documents")
    func addMultipleDocuments() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

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

        try await vectura.reset()
    }

    @Test("Persistence across instances")
    func persistence() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let databaseName = UUID().uuidString
        let config = VecturaConfig(name: "test-db-\(databaseName)")
        let vectura = try await VecturaKit(config: config)

        let texts = ["Document 1", "Document 2"]
        let ids = try await vectura.addDocuments(texts: texts)

        let newVectura = try await VecturaKit(config: config)
        let results = try await newVectura.search(query: "Document")
        #expect(results.count == 2)
        #expect(ids.contains(results[0].id))
        #expect(ids.contains(results[1].id))

        try await vectura.reset()
        try await newVectura.reset()
    }

    @Test("Search threshold reduces results")
    func searchThreshold() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let documents = [
            "Very relevant document about cats",
            "Somewhat relevant about pets",
            "Completely irrelevant about weather"
        ]
        _ = try await vectura.addDocuments(texts: documents)

        let results = try await vectura.search(query: "cats and pets", threshold: 0.8)
        #expect(results.count < 3)

        try await vectura.reset()
    }

    @Test("Custom identifiers are preserved")
    func customIds() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let customId = UUID()
        let text = "Document with custom ID"

        let resultId = try await vectura.addDocument(text: text, id: customId)
        #expect(customId == resultId)

        let results = try await vectura.search(query: text)
        #expect(results.count == 1)
        #expect(results[0].id == customId)

        try await vectura.reset()
    }

    @Test("Model reuse remains performant")
    func modelReuse() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let start = Date()
        for i in 1...5 {
            _ = try await vectura.addDocument(text: "Test document \(i)")
        }
        let duration = Date().timeIntervalSince(start)

        #expect(duration < 10.0)

        try await vectura.reset()
    }

    @Test("Empty search returns no results")
    func emptySearch() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let results = try await vectura.search(query: "test query")
        #expect(results.count == 0, "Search on empty database should return no results")

        try await vectura.reset()
    }

    @Test("Dimension mismatch surfaces error")
    func dimensionMismatch() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let wrongConfig = VecturaConfig(name: "wrong-dim-db-\(UUID().uuidString)", dimension: 128)
        let wrongVectura = try await VecturaKit(config: wrongConfig)

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

        try await wrongVectura.reset()
    }

    @Test("Duplicate identifiers overwrite documents")
    func duplicateIds() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let id = UUID()
        let text1 = "First document"
        let text2 = "Second document"

        _ = try await vectura.addDocument(text: text1, id: id)
        _ = try await vectura.addDocument(text: text2, id: id)

        let results = try await vectura.search(query: text2)
        #expect(results.count == 1)
        #expect(results[0].text == text2)

        try await vectura.reset()
    }

    @Test("Threshold edge cases")
    func searchThresholdEdgeCases() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        _ = try await vectura.addDocuments(texts: ["Test document"])

        let highThresholdResults = try await vectura.search(query: "completely different query", threshold: 0.95)
        #expect(highThresholdResults.count <= 1)

        let allResults = try await vectura.search(query: "completely different", threshold: 0.0)
        #expect(allResults.count >= 0)

        try await vectura.reset()
    }

    @Test("Large number of documents")
    func largeNumberOfDocuments() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

        let documentCount = 100
        let documents = (0..<documentCount).map { "Test document number \($0)" }

        let ids = try await vectura.addDocuments(texts: documents)
        #expect(ids.count == documentCount)

        let results = try await vectura.search(query: "document", numResults: 10)
        #expect(results.count == 10)

        try await vectura.reset()
    }

    @Test("Persistence after reset")
    func persistenceAfterReset() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let databaseName = UUID().uuidString
        let config = VecturaConfig(name: "test-db-\(databaseName)")
        let vectura = try await VecturaKit(config: config)

        let text = "Test document"
        _ = try await vectura.addDocument(text: text)

        try await vectura.reset()

        let results = try await vectura.search(query: text)
        #expect(results.count == 0)

        let newVectura = try await VecturaKit(config: config)
        let newResults = try await newVectura.search(query: text)
        #expect(newResults.count == 0)

        try await newVectura.reset()
    }

    @Test("Folder URL model source")
    func folderURLModelSource() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let config = VecturaConfig(name: "test-db-\(UUID().uuidString)")
        let vectura = try await VecturaKit(config: config)

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

        let ids = try await vectura.addDocuments(texts: documents, model: .folder(url))
        #expect(ids.count == 3)

        let results = try await vectura.search(query: "quick jumping animals", model: .folder(url))
        #expect(results.count >= 2)
        #expect(results[0].score > results[1].score)

        try await vectura.reset()
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
            config: .init(name: databaseName, directoryURL: customDirectoryURL)
        )
        let text = "Test document"
        let id = UUID()
        _ = try await instance.addDocument(text: text, id: id)

        let documentPath = customDirectoryURL
            .appendingPathComponent(databaseName, isDirectory: true)
            .appendingPathComponent("\(id).json")
            .path

        #expect(
            FileManager.default.fileExists(atPath: documentPath),
            "Expected stored document at \(documentPath)"
        )

        try await instance.reset()
    }
}
