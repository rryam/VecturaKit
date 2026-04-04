import Foundation
import VecturaKit
import VecturaOAIKit

@main
struct TestOAIExamples {
    static func main() async throws {
        let environment = ProcessInfo.processInfo.environment
        let baseURL = environment["VECTURA_OAI_BASE_URL"] ?? "http://localhost:1234/v1"
        let model = environment["VECTURA_OAI_MODEL"] ?? ""
        let apiKey = environment["VECTURA_OAI_API_KEY"]

        guard !model.isEmpty else {
            throw NSError(
                domain: "TestOAIExamples",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Set VECTURA_OAI_MODEL before running the example."]
            )
        }

        debugPrint("Testing VecturaOAIKit examples")

        let config = try VecturaConfig(
            name: "test-oai-vector-db"
        )
        let vectorDB = try await VecturaKit(
            config: config,
            embedder: OpenAICompatibleEmbedder(
                baseURL: baseURL,
                model: model,
                apiKey: apiKey,
                timeoutInterval: 120,
                retryAttempts: 2,
                retryBaseDelaySeconds: 1
            )
        )

        let texts = [
            "First document text",
            "Second document text",
            "Third document text",
        ]
        let documentIDs = try await vectorDB.addDocuments(texts: texts)
        debugPrint("Documents added with IDs: \(documentIDs)")

        let results = try await vectorDB.search(
            query: "document text",
            numResults: 5,
            threshold: 0.8
        )

        debugPrint("Search found \(results.count) results:")
        for result in results {
            debugPrint("ID: \(result.id)")
            debugPrint("Text: \(result.text)")
            debugPrint("Score: \(result.score)")
            debugPrint("Created At: \(result.createdAt)")
        }

        if let documentToUpdate = documentIDs.first {
            try await vectorDB.updateDocument(id: documentToUpdate, newText: "Updated text")
        }

        try await vectorDB.reset()
        debugPrint("Database reset")
    }
}
