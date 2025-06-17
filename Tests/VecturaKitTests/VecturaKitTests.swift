import XCTest

@testable import VecturaKit
import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
final class VecturaKitTests: XCTestCase {
    var vectura: VecturaKit!
    var config: VecturaConfig!
    
    override func setUp() async throws {
        config = VecturaConfig(name: "test-db", dimension: 384)
        vectura = try await VecturaKit(config: config)
    }
    
    override func tearDown() async throws {
        try await vectura.reset()
        vectura = nil
    }
    
    func testAddAndSearchDocument() async throws {
        let text = "This is a test document"
        let id = try await vectura.addDocument(text: text)
        
        let results = try await vectura.search(query: "test document")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, id)
        XCTAssertEqual(results[0].text, text)
    }
    
    func testAddMultipleDocuments() async throws {
        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ]
        
        let ids = try await vectura.addDocuments(texts: documents)
        XCTAssertEqual(ids.count, 3)
        
        let results = try await vectura.search(query: "quick jumping animals")
        XCTAssertGreaterThanOrEqual(results.count, 2)
        XCTAssertTrue(results[0].score > results[1].score)
    }
    
    func testPersistence() async throws {
        // Add documents
        let texts = ["Document 1", "Document 2"]
        let ids = try await vectura.addDocuments(texts: texts)
        
        // Create new instance with same config
        let config = VecturaConfig(name: "test-db", dimension: 384)
        let newVectura = try await VecturaKit(config: config)
        
        // Search should work with new instance
        let results = try await newVectura.search(query: "Document")
        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(ids.contains(results[0].id))
        XCTAssertTrue(ids.contains(results[1].id))
    }
    
    func testSearchThreshold() async throws {
        let documents = [
            "Very relevant document about cats",
            "Somewhat relevant about pets",
            "Completely irrelevant about weather",
        ]
        _ = try await vectura.addDocuments(texts: documents)
        
        // With high threshold, should get fewer results
        let results = try await vectura.search(query: "cats and pets", threshold: 0.8)
        XCTAssertLessThan(results.count, 3)
    }
    
    func testCustomIds() async throws {
        let customId = UUID()
        let text = "Document with custom ID"
        
        let resultId = try await vectura.addDocument(text: text, id: customId)
        XCTAssertEqual(customId, resultId)
        
        let results = try await vectura.search(query: text)
        XCTAssertEqual(results[0].id, customId)
    }
    
    func testModelReuse() async throws {
        // Multiple operations should reuse the same model
        let start = Date()
        for i in 1...5 {
            _ = try await vectura.addDocument(text: "Test document \(i)")
        }
        let duration = Date().timeIntervalSince(start)
        
        // If model is being reused, this should be relatively quick
        XCTAssertLessThan(duration, 5.0)  // Adjust threshold as needed
    }
    
    func testEmptySearch() async throws {
        let results = try await vectura.search(query: "test query")
        XCTAssertEqual(results.count, 0, "Search on empty database should return no results")
    }
    
    func testDimensionMismatch() async throws {
        // Test with wrong dimension config
        let wrongConfig = VecturaConfig(name: "wrong-dim-db", dimension: 128)
        let wrongVectura = try await VecturaKit(config: wrongConfig)
        
        let text = "Test document"
        
        do {
            _ = try await wrongVectura.addDocument(text: text)
            XCTFail("Expected dimension mismatch error")
        } catch let error as VecturaError {
            // Should throw dimension mismatch since BERT model outputs 384 dimensions
            switch error {
            case .dimensionMismatch(let expected, let got):
                XCTAssertEqual(expected, 128)
                XCTAssertEqual(got, 384)
            default:
                XCTFail("Wrong error type: \(error)")
            }
        }
    }
    
    func testDuplicateIds() async throws {
        let id = UUID()
        let text1 = "First document"
        let text2 = "Second document"
        
        // Add first document
        _ = try await vectura.addDocument(text: text1, id: id)
        
        // Adding second document with same ID should overwrite
        _ = try await vectura.addDocument(text: text2, id: id)
        
        let results = try await vectura.search(query: text2)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].text, text2)
    }
    
    func testSearchThresholdEdgeCases() async throws {
        let documents = ["Test document"]
        _ = try await vectura.addDocuments(texts: documents)
        
        // Test with threshold = 1.0 (exact match only)
        let perfectResults = try await vectura.search(query: "Test document", threshold: 1.0)
        XCTAssertEqual(perfectResults.count, 0)  // Should find no perfect matches due to encoding differences
        
        // Test with threshold = 0.0 (all matches)
        let allResults = try await vectura.search(query: "completely different", threshold: 0.0)
        XCTAssertEqual(allResults.count, 1)  // Should return all documents
    }
    
    func testLargeNumberOfDocuments() async throws {
        let documentCount = 100
        var documents: [String] = []
        
        for i in 0..<documentCount {
            documents.append("Test document number \(i)")
        }
        
        let ids = try await vectura.addDocuments(texts: documents)
        XCTAssertEqual(ids.count, documentCount)
        
        let results = try await vectura.search(query: "document", numResults: 10)
        XCTAssertEqual(results.count, 10)
    }
    
    func testPersistenceAfterReset() async throws {
        // Add a document
        let text = "Test document"
        _ = try await vectura.addDocument(text: text)
        
        // Reset the database
        try await vectura.reset()
        
        // Verify search returns no results
        let results = try await vectura.search(query: text)
        XCTAssertEqual(results.count, 0)
        
        // Create new instance and verify it's empty
        let newVectura = try await VecturaKit(config: config)
        let newResults = try await newVectura.search(query: text)
        XCTAssertEqual(newResults.count, 0)
    }
    
    func testFolderURLModelSource() async throws {
        /// First load the model from a remote source in order to make it available in the local filesystem.
        _ = try await Bert.loadModelBundle(from: .default)
        
        /// Local model will be downloaded to a predictable location (this may break if `swift-transformers` updates where it downloads models).
        let url = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .appending(path: "huggingface/models/\(VecturaModelSource.defaultModelId)")
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path(percentEncoded: false)), "Expected downloaded model to be available locally at \(url.path())")
        
        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ]
        
        /// Proceed as usual now, but loading the model directly from the local directory instead of downloading it.
        let ids = try await vectura.addDocuments(texts: documents, model: .folder(url))
        XCTAssertEqual(ids.count, 3)
        
        let results = try await vectura.search(query: "quick jumping animals")
        XCTAssertGreaterThanOrEqual(results.count, 2)
        XCTAssertTrue(results[0].score > results[1].score)
    }
    
    func testCustomStorageDirectory() async throws {
        let customDirectoryURL = URL(filePath: NSTemporaryDirectory()).appending(path: "VecturaKitTest")
        defer { try? FileManager.default.removeItem(at: customDirectoryURL) }
        
        let instance = try await VecturaKit(config: .init(name: "test", directoryURL: customDirectoryURL, dimension: 384))
        let text = "Test document"
        let id = UUID()
        _ = try await instance.addDocument(text: text, id: id)
        
        let documentPath = customDirectoryURL.appending(path: "test/\(id).json").path(percentEncoded: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: documentPath), "Custom storage directory inserted document doesn't exist at \(documentPath)")
    }
}
