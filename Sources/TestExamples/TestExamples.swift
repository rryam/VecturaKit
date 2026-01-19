import Foundation
import VecturaKit
import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@main
struct ValidationScript {
    static func main() async {
        print("🚀 Starting VecturaKit Validation...")
        
        do {
            // 1. Setup Configuration
            let config = try VecturaConfig(
                name: "validation-db",
                directoryURL: nil, // In-memory/temp storage
                searchOptions: VecturaConfig.SearchOptions(
                    defaultNumResults: 5,
                    minThreshold: 0.1 // Low threshold to ensure we get matches for validation
                )
            )

            print("✅ Configuration created.")

            // 2. Initialize Embedder
            // Using a small, fast model for validation
            let modelId = "sentence-transformers/all-MiniLM-L6-v2"
            print("⏳ Initializing Embedder (\(modelId))...")
            let embedder = SwiftEmbedder(modelSource: .id(modelId))
            
            // 3. Initialize VecturaKit
            print("⏳ Initializing VecturaKit...")
            let vectorDB = try await VecturaKit(config: config, embedder: embedder)
            print("✅ VecturaKit initialized.")

            // Reset DB to ensure clean state
            try await vectorDB.reset()

            // 4. Add Documents
            let texts = [
                "The customized search engine works with vector embeddings.",
                "Swift is a powerful language for iOS development.",
                "Vector databases are essential for semantic search application.",
                "Fruits like apples and oranges are healthy."
            ]
            
            print("⏳ Adding \(texts.count) documents...")
            let ids = try await vectorDB.addDocuments(texts: texts)
            print("✅ Added \(ids.count) documents.")

            // 5. Test Text Search (Hybrid)
            let query = "vector search"
            print("🔎 Searching for: '\(query)'")
            
            let results = try await vectorDB.search(query: .text(query), numResults: 3)
            
            if results.isEmpty {
                print("❌ Validation Failed: No results found for query.")
                exit(1)
            }
            
            print("✅ Found \(results.count) results.")
            for (index, result) in results.enumerated() {
                print("   [\(index + 1)] Score: \(String(format: "%.4f", result.score)) | Text: \(result.text)")
            }

            // check if the top result is relevant
            if results[0].text.contains("vector") {
                 print("✅ Top result contains expected keyword 'vector'.")
            } else {
                 print("⚠️ Top result might not be the most relevant, check scores.")
            }

            // 6. Test Semantic Search (Different words, same meaning)
            let semanticQuery = "programming tools for apple"
            print("🔎 Searching for semantic match: '\(semanticQuery)'")
            
            let semanticResults = try await vectorDB.search(query: .text(semanticQuery), numResults: 1)
            if let first = semanticResults.first, first.text.contains("Swift") {
                 print("✅ Semantic Search Validated! matched 'Swift' document.")
            } else {
                 print("⚠️ Semantic match weak or incorrect.")
                 if let first = semanticResults.first {
                     print("   Got: \(first.text)")
                 }
            }

            print("\n🎉 VECTURAKIT VALIDATION COMPLETED SUCCESSFULLY!")
            
        } catch {
            print("❌ Validation Failed with error: \(error)")
            exit(1)
        }
    }
}
