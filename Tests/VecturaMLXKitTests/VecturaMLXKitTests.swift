//
//  VecturaMLXKitTests.swift
//  VecturaKit
//
//  Created by Rudrank Riyam on 2/1/25.
//

import XCTest
import Foundation
@testable import VecturaMLXKit
@testable import VecturaKit

// Assuming VecturaConfig and its searchOptions are defined similar to the following:
// You can remove this extension or adjust it if your actual VecturaConfig already provides a suitable initializer.
extension VecturaConfig {
    init(name: String, directoryURL: URL, minThreshold: Float = 0, defaultNumResults: Int = 10) {
        // initializer implementation may vary – adjust accordingly.
        // Here we assume VecturaConfig is Codable.
        // For testing, we just supply the required values.
        self.init(name: name, directoryURL: directoryURL, dimension: 786)
    }
}

final class VecturaMLXKitTests: XCTestCase {
    
    var testDirectory: URL!
    
    override func setUpWithError() throws {
        // Use a temporary directory for testing.
        let temp = FileManager.default.temporaryDirectory
        testDirectory = temp.appendingPathComponent("VecturaMLXKitTests", isDirectory: true)
        if FileManager.default.fileExists(atPath: testDirectory.path) {
            try FileManager.default.removeItem(at: testDirectory)
        }
        try FileManager.default.createDirectory(at: testDirectory, withIntermediateDirectories: true)
    }
    
    override func tearDownWithError() throws {
        // Clean up the test directory
        if FileManager.default.fileExists(atPath: testDirectory.path) {
            try FileManager.default.removeItem(at: testDirectory)
        }
    }
    
    func testAddAndSearch() async throws {
        // Create a test config which sets minThreshold to 0 so that any document is returned.
        let config = VecturaConfig(name: "TestDB", directoryURL: testDirectory, minThreshold: 0, defaultNumResults: 10)
        let kit = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        
        let text = "Hello world"
        let ids = try await kit.addDocuments(texts: [text])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        // Search using a query string that exactly matches the text.
        let results = try await kit.search(query: text)
        XCTAssertEqual(results.count, 1, "The search should return one result after adding one document.")
        XCTAssertEqual(results.first?.text, text, "The text of the found document should match the added text.")
    }
    
    func testDeleteDocuments() async throws {
        let config = VecturaConfig(name: "TestDB", directoryURL: testDirectory, minThreshold: 0, defaultNumResults: 10)
        let kit = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        
        let text = "Delete me"
        let ids = try await kit.addDocuments(texts: [text])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        // Now delete the document
        try await kit.deleteDocuments(ids: ids)
        
        let results = try await kit.search(query: text)
        XCTAssertTrue(results.isEmpty, "After deletion, the document should not be found.")
    }
    
    func testUpdateDocument() async throws {
        let config = VecturaConfig(name: "TestDB", directoryURL: testDirectory, minThreshold: 0, defaultNumResults: 10)
        let kit = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        
        let originalText = "Original text"
        let updatedText = "Updated text"
        let ids = try await kit.addDocuments(texts: [originalText])
        XCTAssertEqual(ids.count, 1, "Should add exactly one document.")
        
        let documentID = ids.first!
        try await kit.updateDocument(id: documentID, newText: updatedText)
        
        let results = try await kit.search(query: updatedText)
        XCTAssertEqual(results.count, 1, "There should be one document after update.")
        XCTAssertEqual(results.first?.text, updatedText, "The document text should be updated.")
    }
    
    func testReset() async throws {
        let config = VecturaConfig(name: "TestDB", directoryURL: testDirectory, minThreshold: 0, defaultNumResults: 10)
        let kit = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        
        // Add multiple documents
        _ = try await kit.addDocuments(texts: ["Doc1", "Doc2"])
        // Reset the store
        try await kit.reset()
        
        let results = try await kit.search(query: "Doc")
        XCTAssertTrue(results.isEmpty, "After a reset, the search should return no results.")
    }
}
