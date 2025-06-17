import XCTest
import Foundation
import Metal
import MLX
@testable import VecturaMLXKit
@testable import VecturaKit

/// Tests for VecturaMLXKit functionality
/// 
/// Note: These tests require:
/// 1. Metal device (GPU) availability
/// 2. Metal Toolchain (install via: xcodebuild -downloadComponent MetalToolchain)
/// 3. MLX device libraries to be available
/// 
/// Run tests with: xcodebuild test -scheme VecturaMLXKitTests -destination 'platform=macOS'
/// (swift test may not work due to Metal library compilation requirements)
@available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, *)
final class VecturaMLXKitTests: XCTestCase {
    
    var testDirectory: URL!
    // Set a dimension matching your model expectation (e.g., 768)
    let testDimension = 768
    
    // Helper method to check Metal and MLX availability and create VecturaMLXKit safely
    private func createVecturaMLXKit(config: VecturaConfig) async throws -> VecturaMLXKit {
        // Check Metal device availability first to avoid crashes
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available. MLX requires Metal for GPU acceleration.")
        }
        
        do {
            return try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        } catch {
            // If model loading fails (common in CI/test environments), skip the test
            throw XCTSkip("Failed to initialize VecturaMLXKit: \(error.localizedDescription). This may be due to Metal/GPU unavailability, missing Metal Toolchain, or MLX device library issues in test environment.")
        }
    }
    
    override func setUpWithError() throws {
        // Create a temporary directory for testing with unique identifier to avoid conflicts
        let temp = FileManager.default.temporaryDirectory
        let uniqueId = UUID().uuidString.prefix(8)
        testDirectory = temp.appendingPathComponent("VecturaMLXKitTests-\(uniqueId)", isDirectory: true)
        
        // Clean up any existing directory
        if FileManager.default.fileExists(atPath: testDirectory.path) {
            try FileManager.default.removeItem(at: testDirectory)
        }
        
        // Create the directory with proper permissions
        try FileManager.default.createDirectory(
            at: testDirectory, 
            withIntermediateDirectories: true, 
            attributes: [.posixPermissions: 0o755]
        )
    }
    
    override func tearDownWithError() throws {
        // Clean up the temporary directory.
        if let testDirectory = testDirectory, 
           FileManager.default.fileExists(atPath: testDirectory.path) {
            do {
                try FileManager.default.removeItem(at: testDirectory)
            } catch {
                // Don't fail the test if cleanup fails, just log it
                print("Warning: Failed to clean up test directory: \(error)")
            }
        }
        testDirectory = nil
    }
    
    func testAddAndSearch() async throws {
        // Create a test config with a minThreshold of 0 so any document is returned.
        let config = try VecturaConfig(
            name: "TestDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        
        let kit = try await createVecturaMLXKit(config: config)
        
        let text = "Hello world"
        let ids = try await kit.addDocuments(texts: [text])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        // Perform a search using the same text.
        let results = try await kit.search(query: text)
        XCTAssertEqual(results.count, 1, "The search should return one result after adding one document.")
        XCTAssertEqual(results.first?.text, text, "The text of the returned document should match the added text.")
    }
    
    func testDeleteDocuments() async throws {
        let config = try VecturaConfig(
            name: "TestDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        let text = "Delete me"
        let ids = try await kit.addDocuments(texts: [text])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        try await kit.deleteDocuments(ids: ids)
        
        let results = try await kit.search(query: text)
        XCTAssertTrue(results.isEmpty, "After deletion, the document should not be returned in search results.")
    }
    
    func testUpdateDocument() async throws {
        let config = try VecturaConfig(
            name: "TestDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        let originalText = "Original text"
        let updatedText = "Updated text"
        let ids = try await kit.addDocuments(texts: [originalText])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        let documentID = ids.first!
        try await kit.updateDocument(id: documentID, newText: updatedText)
        
        let results = try await kit.search(query: updatedText)
        XCTAssertEqual(results.count, 1, "One document should be returned after update.")
        XCTAssertEqual(results.first?.text, updatedText, "The document text should be updated in the search results.")
    }
    
    func testReset() async throws {
        let config = try VecturaConfig(
            name: "TestDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        _ = try await kit.addDocuments(texts: ["Doc1", "Doc2"])
        try await kit.reset()
        
        let results = try await kit.search(query: "Doc")
        XCTAssertTrue(results.isEmpty, "After a reset, search should return no results.")
    }
    
    // MARK: - Robust Search Tests
    
    func testSearchMultipleDocuments() async throws {
        let config = try VecturaConfig(
            name: "TestMLXDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        // Add several documents with overlapping keywords.
        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over lazy hounds",
            "An agile brown fox",
            "Lazy dogs sleep all day",
            "Quick and nimble foxes"
        ]
        _ = try await kit.addDocuments(texts: texts)
        
        // Search for an expression close to "brown fox".
        let results = try await kit.search(query: "brown fox")
        
        // We expect at least two results related to 'brown fox'.
        XCTAssertGreaterThanOrEqual(results.count, 2, "Should return at least two documents related to 'brown fox'.")
        
        // Verify that results are sorted in descending order by score.
        for i in 1..<results.count {
            XCTAssertGreaterThanOrEqual(results[i - 1].score, results[i].score, "Search results are not sorted in descending order by score.")
        }
    }
    
    func testSearchNumResultsLimiting() async throws {
        let config = try VecturaConfig(
            name: "TestMLXDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        // Add more documents.
        let texts = [
            "Document one about testing",
            "Document two about testing",
            "Document three about testing",
            "Document four about testing",
            "Document five about testing"
        ]
        _ = try await kit.addDocuments(texts: texts)
        
        // Request only 3 results.
        let results = try await kit.search(query: "testing", numResults: 3)
        XCTAssertEqual(results.count, 3, "Should limit the search results to exactly 3 documents.")
    }
    
    func testSearchWithHighThreshold() async throws {
        // Set a high threshold so that only nearly identical matches return.
        let config = try VecturaConfig(
            name: "TestMLXDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        // Add documents that are expected to have high similarity for 'apple'.
        let texts = [
            "Apple pie recipe",
            "Delicious apple tart",
            "Banana bread instructions"
        ]
        _ = try await kit.addDocuments(texts: texts)
        
        // Use a high threshold (e.g., 0.99) to filter out less-similar documents.
        let highThreshold: Float = 0.99
        let results = try await kit.search(query: "apple", threshold: highThreshold)
        
        // Verify that all returned documents have a similarity score meeting or exceeding the threshold.
        for result in results {
            XCTAssertGreaterThanOrEqual(result.score, highThreshold, "Result score \(result.score) is below the high threshold \(highThreshold).")
        }
    }
    
    func testSearchNoMatches() async throws {
        let config = try VecturaConfig(
            name: "TestMLXDB",
            directoryURL: testDirectory,
            dimension: testDimension,
            searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10, minThreshold: 0, hybridWeight: 0.5, k1: 1.2, b: 0.75)
        )
        let kit = try await createVecturaMLXKit(config: config)
        
        // Add a document.
        _ = try await kit.addDocuments(texts: ["Some random content"])
        
        // Use a query that should not match with a high threshold.
        let results = try await kit.search(query: "completely different query text", threshold: 0.9)
        XCTAssertTrue(results.isEmpty, "Search should return no results when the query does not match any document.")
    }
}
