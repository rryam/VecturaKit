// Test script for VecturaMLXKit README examples
import Foundation
import VecturaKit
import VecturaMLXKit
import MLXEmbedders

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
@main
struct TestMLXExamples {
    static func main() async throws {
        debugPrint("Testing VecturaMLXKit Examples")

        // 2. Initialize Database
        debugPrint("2. Initialize Database")

        let config = VecturaConfig(
          name: "test-mlx-vector-db",
          dimension: 768 //  nomic_text_v1_5 model outputs 768-dimensional embeddings
        )
        let vectorDB = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
        debugPrint("MLX Database initialized successfully")
        debugPrint("Document count: \(vectorDB.documentCount)")

        // 3. Add Documents
        debugPrint("3. Add Documents")

        let texts = [
          "First document text",
          "Second document text",
          "Third document text"
        ]
        let documentIds = try await vectorDB.addDocuments(texts: texts)
        debugPrint("Documents added with IDs: \(documentIds)")
        debugPrint("Total document count: \(vectorDB.documentCount)")

        // 4. Search Documents
        debugPrint("4. Search Documents")

        let results = try await vectorDB.search(
            query: "document text",
            numResults: 5,      // Optional
            threshold: 0.8     // Optional
        )

        debugPrint("Search found \(results.count) results:")
        for result in results {
            debugPrint("ID: \(result.id)")
            debugPrint("Text: \(result.text)")
            debugPrint("Score: \(result.score)")
            debugPrint("Created At: \(result.createdAt)")
        }

        // 5. Document Management
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
