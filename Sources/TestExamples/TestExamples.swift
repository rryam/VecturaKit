import Foundation
import VecturaKit
import VecturaNLKit

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
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("VecturaKit-TestExamples-\(UUID().uuidString)", isDirectory: true)

    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: directory) }

      let config = try VecturaConfig(
        name: "validation-db",
        directoryURL: directory,
        searchOptions: VecturaConfig.SearchOptions(
          defaultNumResults: 5,
          minThreshold: 0.1
        )
      )

      let embedder = try await NLContextualEmbedder(language: .english)
      let vectorDB = try await VecturaKit(config: config, embedder: embedder)
      try await vectorDB.reset()

      let texts = [
        "VecturaKit combines vector similarity with BM25 text search for hybrid retrieval.",
        "Swift is the primary language for building apps on Apple platforms like iOS and macOS.",
        "Vector databases store embeddings to power semantic search over text.",
        "On-device search keeps user data private and responsive.",
      ]

      let ids = try await vectorDB.addDocuments(texts: texts)
      guard ids.count == texts.count else {
        throw ValidationError.message("Document count mismatch: expected \(texts.count), got \(ids.count).")
      }

      let hybridResults = try await vectorDB.search(query: .text("hybrid search"), numResults: 3)
      try validateResults(label: "Hybrid search", results: hybridResults, expectedSubstring: "BM25")

      let semanticResults = try await vectorDB.search(query: .text("Apple platform development"), numResults: 3)
      try validateResults(label: "Semantic search", results: semanticResults, expectedSubstring: "Swift")

      let vectorEmbedding = try await embedder.embed(text: "semantic search with embeddings")
      let vectorResults = try await vectorDB.search(query: .vector(vectorEmbedding), numResults: 3)
      try validateResults(label: "Vector search", results: vectorResults, expectedSubstring: "embeddings")

      print("VecturaKit NL examples completed successfully.")
    } catch {
      print("Validation failed: \(error)")
      exit(1)
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
      throw ValidationError.message("\(label): expected a result containing '\(expectedSubstring)'.")
    }
  }
}
