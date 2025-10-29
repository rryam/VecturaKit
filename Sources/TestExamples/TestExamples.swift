// Test script for VecturaKit README examples
import Foundation
import VecturaKit
import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@main
struct TestExamples {
    static func main() async throws {
        debugPrint("Testing Core VecturaKit Examples")

        // Example 2: Create Configuration and Initialize Database
        debugPrint("2. Create Configuration and Initialize Database")
        let config = VecturaConfig(
            name: "test-vector-db",
            directoryURL: nil,  // Optional custom storage location
            // Dimension will be auto-detected from the model
            searchOptions: VecturaConfig.SearchOptions(
                defaultNumResults: 10,
                minThreshold: 0.7,
                hybridWeight: 0.5,  // Balance between vector and text search
                k1: 1.2,           // BM25 parameters
                b: 0.75
            )
        )

        let vectorDB = try await VecturaKit(
            config: config,
            embedder: SwiftEmbedder(modelSource: .id("sentence-transformers/all-MiniLM-L6-v2"))
        )
        debugPrint("Database initialized successfully")
        debugPrint("no space here and use debugPrint everywhere")
        debugPrint("Document count: \(vectorDB.documentCount)")

        // Example 3: Add Documents
        debugPrint("3. Add Documents")

        // Single document:
        debugPrint("Adding single document...")
        let text = "Sample text to be embedded"
        let documentId = try await vectorDB.addDocument(
            text: text,
            id: UUID()  // Optional, will be generated if not provided
        )
        debugPrint("Single document added with ID: \(documentId)")
        debugPrint("Document count: \(vectorDB.documentCount)")

        // Multiple documents in batch:
        debugPrint("Adding multiple documents in batch...")
        let texts = [
            "First document text",
            "Second document text",
            "Third document text"
        ]
        let documentIds = try await vectorDB.addDocuments(
            texts: texts,
            ids: nil  // Optional array of UUIDs
        )
        debugPrint("Batch documents added with IDs: \(documentIds)")
        debugPrint("Total document count: \(vectorDB.documentCount)")

        // Example 4: Search Documents
        debugPrint("4. Search Documents")

        // Search by text (hybrid search):
        debugPrint("Searching by text (hybrid search)...")
        let textResults = try await vectorDB.search(
            query: "document text",
            numResults: 5,      // Optional
            threshold: 0.8     // Optional
        )

        debugPrint("Text search found \(textResults.count) results:")
        for result in textResults {
            debugPrint("ID: \(result.id)")
            debugPrint("Text: \(result.text)")
            debugPrint("Score: \(result.score)")
            debugPrint("Created: \(result.createdAt)")
        }

        // Search by vector embedding:
        debugPrint("Searching by vector embedding...")
        // Use a simple test vector (zeros) for demonstration
        var testVector = [Float](repeating: 0.0, count: 384)
        testVector[0] = 1.0 // Make it slightly different

        let vectorResults = try await vectorDB.search(
            query: testVector,  // [Float] matching config.dimension
            numResults: 5,  // Optional
            threshold: 0.0  // Optional - lower threshold for test vector
        )

        debugPrint("Vector search found \(vectorResults.count) results:")
        for result in vectorResults {
            debugPrint("ID: \(result.id)")
            debugPrint("Text: \(result.text)")
            debugPrint("Score: \(result.score)")
        }

        // Example 5: Document Management
        debugPrint("5. Document Management")

        // Update document:
        debugPrint("Updating document...")
        let documentToUpdate = documentIds.first!
        try await vectorDB.updateDocument(
            id: documentToUpdate,
            newText: "Updated text"
        )
        debugPrint("Document updated")

        // Verify update by searching
        let updatedResults = try await vectorDB.search(query: "Updated text", threshold: 0.0)
        debugPrint("Verification: Found \(updatedResults.count) documents with 'Updated text'")

        // Delete documents:
        debugPrint("Deleting documents...")
        try await vectorDB.deleteDocuments(ids: [documentToUpdate, documentIds[1]])
        debugPrint("Documents deleted")
        debugPrint("Document count after deletion: \(vectorDB.documentCount)")

        // Reset database:
        debugPrint("Resetting database...")
        try await vectorDB.reset()
        debugPrint("Database reset")
        debugPrint("Document count after reset: \(vectorDB.documentCount)")
    }
}
