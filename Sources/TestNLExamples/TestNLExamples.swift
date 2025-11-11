// Test script for VecturaNLKit README examples
import Foundation
import VecturaKit
import VecturaNLKit

enum ExampleError: Error, CustomStringConvertible {
  case noDocumentsToUpdate

  var description: String {
    switch self {
    case .noDocumentsToUpdate:
      return "No documents available to update"
    }
  }
}

@main
struct TestNLExamples {
  static func main() async throws {
    try await initializeEmbedder()
    let embedder = try await NLContextualEmbedder(language: .english)
    let vectorDB = try await initializeDatabase(embedder: embedder)
    let documentIds = try await addDocuments(to: vectorDB)
    try await searchDocuments(in: vectorDB)
    try await testContextualUnderstanding(in: vectorDB)
    try await manageDocuments(in: vectorDB, documentIds: documentIds)
    try await testEmbeddings(embedder: embedder)
    try await resetDatabase(vectorDB)

    debugPrint("\n✅ All VecturaNLKit examples completed successfully!")
  }

  private static func initializeEmbedder() async throws {
    debugPrint("1. Initialize NLContextualEmbedder")

    let embedder = try await NLContextualEmbedder(language: .english)
    let dimension = try await embedder.dimension
    debugPrint("NLContextualEmbedder initialized successfully")
    debugPrint("Embedding dimension: \(dimension)")

    let modelInfo = await embedder.modelInfo
    debugPrint("Model language: \(modelInfo.language.rawValue)")
    if let dimension = modelInfo.dimension {
      debugPrint("Model dimension: \(dimension)")
    }
  }

  private static func initializeDatabase(embedder: NLContextualEmbedder) async throws -> VecturaKit {
    debugPrint("\n2. Initialize Database")

    let config = try VecturaConfig(name: "test-nl-vector-db")
    let vectorDB = try await VecturaKit(
      config: config,
      embedder: embedder
    )
    debugPrint("NL Database initialized successfully")
    debugPrint("Document count: \(try await vectorDB.documentCount)")

    return vectorDB
  }

  private static func addDocuments(to vectorDB: VecturaKit) async throws -> [UUID] {
    debugPrint("\n3. Add Documents")

    let texts = [
      "Natural language understanding is fascinating",
      "Swift makes iOS development enjoyable",
      "Machine learning on device preserves privacy",
      "Vector databases enable semantic search"
    ]
    let documentIds = try await vectorDB.addDocuments(texts: texts)
    debugPrint("Documents added with IDs: \(documentIds)")
    debugPrint("Total document count: \(try await vectorDB.documentCount)")

    return documentIds
  }

  private static func searchDocuments(in vectorDB: VecturaKit) async throws {
    debugPrint("\n4. Search Documents")

    let results = try await vectorDB.search(
      query: "iOS programming",
      numResults: 5,      // Optional
      threshold: 0.7      // Optional
    )

    debugPrint("Search found \(results.count) results:")
    for result in results {
      debugPrint("ID: \(result.id)")
      debugPrint("Text: \(result.text)")
      debugPrint("Score: \(result.score)")
      debugPrint("Created At: \(result.createdAt)")
      debugPrint("---")
    }
  }

  private static func testContextualUnderstanding(in vectorDB: VecturaKit) async throws {
    debugPrint("\n5. Test Contextual Understanding")

    let semanticResults = try await vectorDB.search(
      query: "building apps for Apple platforms",
      numResults: 3,
      threshold: 0.6
    )

    debugPrint("Semantic search found \(semanticResults.count) results:")
    for result in semanticResults {
      debugPrint("Text: \(result.text)")
      debugPrint("Score: \(result.score)")
      debugPrint("---")
    }
  }

  private static func manageDocuments(in vectorDB: VecturaKit, documentIds: [UUID]) async throws {
    debugPrint("\n6. Document Management")

    guard let documentToUpdate = documentIds.first else {
      throw ExampleError.noDocumentsToUpdate
    }

    debugPrint("Updating document...")
    try await vectorDB.updateDocument(
      id: documentToUpdate,
      newText: "Apple's frameworks enable powerful on-device AI"
    )
    debugPrint("Document updated")

    // Verify update by searching
    let updatedResults = try await vectorDB.search(
      query: "on-device AI",
      threshold: 0.6
    )
    debugPrint("Verification: Found \(updatedResults.count) documents related to 'on-device AI'")
    if let first = updatedResults.first {
      debugPrint("Top result: \(first.text)")
    }

    debugPrint("\nDeleting documents...")
    let idsToDelete = documentIds.count >= 2
      ? [documentToUpdate, documentIds[1]]
      : [documentToUpdate]
    try await vectorDB.deleteDocuments(ids: idsToDelete)
    debugPrint("Documents deleted")
    debugPrint("Document count after deletion: \(try await vectorDB.documentCount)")
  }

  private static func testEmbeddings(embedder: NLContextualEmbedder) async throws {
    debugPrint("\n7. Test Single Embedding")

    let singleText = "Testing NLContextualEmbedding"
    let singleEmbedding = try await embedder.embed(text: singleText)
    debugPrint("Generated embedding for: '\(singleText)'")
    debugPrint("Embedding length: \(singleEmbedding.count)")
    debugPrint("First 5 values: \(Array(singleEmbedding.prefix(5)))")

    debugPrint("\n8. Test Batch Embedding")

    let batchTexts = [
      "First test",
      "Second test",
      "Third test"
    ]
    let batchEmbeddings = try await embedder.embed(texts: batchTexts)
    debugPrint("Generated \(batchEmbeddings.count) embeddings")
    for (index, embedding) in batchEmbeddings.enumerated() {
      debugPrint("Embedding \(index + 1): length = \(embedding.count)")
    }
  }

  private static func resetDatabase(_ vectorDB: VecturaKit) async throws {
    debugPrint("\nResetting database...")
    try await vectorDB.reset()
    debugPrint("Database reset")
    debugPrint("Document count after reset: \(try await vectorDB.documentCount)")
  }
}
