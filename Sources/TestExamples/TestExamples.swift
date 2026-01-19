import Foundation
import VecturaKit
import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@main
struct ValidationScript {
    enum ValidationError: Error, CustomStringConvertible {
        case message(String)

        var description: String {
            switch self {
            case .message(let text):
                return text
            }
        }
    }

    static func main() async {
        print("Starting VecturaKit validation.")

        do {
            let config = try VecturaConfig(
                name: "validation-db",
                directoryURL: nil,
                searchOptions: VecturaConfig.SearchOptions(
                    defaultNumResults: 5,
                    minThreshold: 0.1
                )
            )

            print("Configuration created.")

            let embedder = SwiftEmbedder(modelSource: .default)
            print("Embedder initialized with the default model.")

            let vectorDB = try await VecturaKit(config: config, embedder: embedder)
            print("VecturaKit initialized.")

            try await vectorDB.reset()

            let texts = [
                "VecturaKit combines vector similarity with BM25 text search for hybrid retrieval.",
                "Swift is the primary language for building apps on Apple platforms like iOS and macOS.",
                "Vector databases store embeddings to power semantic search over text.",
                "On-device search keeps user data private and responsive."
            ]

            print("Adding \(texts.count) documents...")
            let ids = try await vectorDB.addDocuments(texts: texts)
            guard ids.count == texts.count else {
                throw ValidationError.message("Document count mismatch: expected \(texts.count), got \(ids.count).")
            }
            print("Documents added: \(ids.count).")

            let hybridQuery = "hybrid search"
            let hybridResults = try await vectorDB.search(query: .text(hybridQuery), numResults: 3)
            logResults(title: "Hybrid search", query: hybridQuery, results: hybridResults)
            try validateResults(
                label: "Hybrid search",
                results: hybridResults,
                expectedSubstring: "BM25"
            )

            let semanticQuery = "Apple platform development"
            let semanticResults = try await vectorDB.search(query: .text(semanticQuery), numResults: 3)
            logResults(title: "Semantic search", query: semanticQuery, results: semanticResults)
            try validateResults(
                label: "Semantic search",
                results: semanticResults,
                expectedSubstring: "Swift"
            )

            let vectorQuery = "semantic search with embeddings"
            let vectorEmbedding = try await embedder.embed(text: vectorQuery)
            let vectorResults = try await vectorDB.search(
                query: .vector(vectorEmbedding),
                numResults: 3
            )
            logResults(title: "Vector search", query: vectorQuery, results: vectorResults)
            try validateResults(
                label: "Vector search",
                results: vectorResults,
                expectedSubstring: "embeddings"
            )

            print("VecturaKit validation completed successfully.")
        } catch {
            print("Validation failed: \(error)")
            exit(1)
        }
    }

    private static func logResults(
        title: String,
        query: String,
        results: [VecturaSearchResult]
    ) {
        print("\n\(title) results for '\(query)': \(results.count)")
        for (index, result) in results.enumerated() {
            let score = String(format: "%.4f", result.score)
            print("  [\(index + 1)] Score: \(score) | Text: \(result.text)")
        }
    }

    private static func validateResults(
        label: String,
        results: [VecturaSearchResult],
        expectedSubstring: String
    ) throws {
        guard !results.isEmpty else {
            throw ValidationError.message("\(label): no results returned.")
        }

        let hasExpected = results.prefix(3).contains { result in
            result.text.localizedCaseInsensitiveContains(expectedSubstring)
        }

        guard hasExpected else {
            throw ValidationError.message(
                "\(label): expected a result containing '\(expectedSubstring)'."
            )
        }
    }
}
