// Test script for VecturaMLXKit README examples
import Foundation
import VecturaKit
import VecturaMLXKit
import MLXEmbedders

@main
struct TestMLXExamples {
  static func main() async throws {
    debugPrint("Testing VecturaMLXKit Examples")

    // 2. Initialize Database
    debugPrint("2. Initialize Database")

    let config = try VecturaConfig(
      name: "test-mlx-vector-db"
      // Dimension will be auto-detected from the model
    )
    let vectorDB = try await VecturaKit(
      config: config,
      embedder: try await MLXEmbedder(configuration: .nomic_text_v1_5)
    )
    debugPrint("MLX Database initialized successfully")
    debugPrint("Document count: \(try await vectorDB.documentCount)")

    // 3. Add Documents
    debugPrint("3. Add Documents")

    let texts = [
      "First document text",
      "Second document text",
      "Third document text"
    ]
    let documentIds = try await vectorDB.addDocuments(texts: texts)
    debugPrint("Documents added with IDs: \(documentIds)")
    debugPrint("Total document count: \(try await vectorDB.documentCount)")

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
    guard let documentToUpdate = documentIds.first else {
      debugPrint("No documents to update")
      return
    }
    debugPrint("Updating document...")
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
    let idsToDelete = documentIds.count >= 2
      ? [documentToUpdate, documentIds[1]]
      : [documentToUpdate]
    try await vectorDB.deleteDocuments(ids: idsToDelete)
    debugPrint("Documents deleted")
    debugPrint("Document count after deletion: \(try await vectorDB.documentCount)")

    // Reset database:
    debugPrint("Resetting database...")
    try await vectorDB.reset()
    debugPrint("Database reset")
    debugPrint("Document count after reset: \(try await vectorDB.documentCount)")
  }
}
