import Testing
import MLX
import MLXEmbedders
@testable import VecturaKit

@Suite("VecturaKit Tests")
final class VecturaKitTests {
    @Test("Add and retrieve document")
    func testAddDocument() async throws {
        let config = VecturaConfig(name: "test_db", dimension: 3)
        let db = try VecturaKit(config: config)

        let testEmbedding = MLXArray(converting: [1.0, 2.0, 3.0])
        let testText = "Test document"

        let docId = try await db.addDocument(text: testText, embedding: testEmbedding)
        let searchResults = try await db.search(query: testEmbedding, numResults: 1)

        #expect(searchResults.count == 1)
        #expect(searchResults[0].id == docId)
        #expect(searchResults[0].text == testText)
        #expect(searchResults[0].score > 0.99) // Should be very similar to itself
    }

    @Test("Search with threshold")
    func testSearchWithThreshold() async throws {
        let config = VecturaConfig(name: "test_db", dimension: 2)
        let db = try VecturaKit(config: config)

        // Add two documents with different directions
        let doc1 = MLXArray(converting: [1.0, 0.0])
        let doc2 = MLXArray(converting: [0.0, 1.0])

        _ = try await db.addDocument(text: "Document 1", embedding: doc1)
        _ = try await db.addDocument(text: "Document 2", embedding: doc2)

        // Search with a threshold that should only return one document
        let query = MLXArray(converting: [0.9, 0.1])
        let results = try await db.search(query: query, threshold: 0.8)

        #expect(results.count == 1)
        #expect(results[0].text == "Document 1")
    }

    @Test("Reset database")
    func testReset() async throws {
        let config = VecturaConfig(name: "test_db", dimension: 2)
        let db = try VecturaKit(config: config)

        let embedding = MLXArray(converting: [1.0, 1.0])
        _ = try await db.addDocument(text: "Test", embedding: embedding)

        try await db.reset()

        let results = try await db.search(query: embedding)
        #expect(results.isEmpty)
    }
}
