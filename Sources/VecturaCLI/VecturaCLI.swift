import ArgumentParser
import Foundation
import VecturaKit

// MARK: - Data Models for Mock Dataset

struct MockDataset: Decodable {
  struct Category: Decodable {
    let name: String
    let documents: [String]
  }
  let categories: [Category]

  var totalDocuments: Int {
    categories.reduce(0) { $0 + $1.documents.count }
  }

  var allDocuments: [String] {
    categories.flatMap { $0.documents }
  }
}

// MARK: - Helper Extensions

extension String {
  static func * (left: String, right: Int) -> String {
    return String(repeating: left, count: right)
  }
}

extension Duration {
  var timeInterval: TimeInterval {
    let (seconds, attoseconds) = self.components
    return TimeInterval(seconds) + (TimeInterval(attoseconds) / 1e18)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@main
struct VecturaCLI: AsyncParsableCommand {
  struct DocumentID: ExpressibleByArgument, Decodable {
    let uuid: UUID

    init(_ uuid: UUID) {
      self.uuid = uuid
    }

    init?(argument: String) {
      guard let uuid = UUID(uuidString: argument) else { return nil }
      self.uuid = uuid
    }
  }

  static let configuration = CommandConfiguration(
    commandName: "vectura",
    abstract: "A CLI tool for VecturaKit vector database",
    subcommands: [Add.self, Search.self, Update.self, Delete.self, Reset.self, Mock.self]
  )

  /// Writes an error message to stderr
  static func writeError(_ message: String) {
    let errorMessage = message + "\n"
    if let data = errorMessage.data(using: .utf8) {
      FileHandle.standardError.write(data)
    }
  }

  static func setupDB(dbName: String, dimension: Int, numResults: Int, threshold: Float, modelId: String) async throws
  -> VecturaKit {
    let config = try VecturaConfig(
      name: dbName,
      dimension: dimension,
      searchOptions: VecturaConfig.SearchOptions(
        defaultNumResults: numResults,
        minThreshold: threshold
      )
    )
    let embedder = SwiftEmbedder(modelSource: .id(modelId))
    return try await VecturaKit(config: config, embedder: embedder)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension VecturaCLI {
  struct Mock: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Run a demonstration with 1000+ sample documents showcasing semantic search"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-demo-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 512

    @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold for searches")
    var threshold: Float = 0.5

    @Option(name: [.long, .customShort("n")], help: "Number of results to return per search")
    var numResults: Int = 5

    @Option(name: [.long, .customShort("m")], help: "Model ID for embeddings")
    var modelId: String = "minishlab/potion-retrieval-32M"

    mutating func run() async throws {
      print("VecturaKit Mock Demonstration")
      print("=" * 60)

      // Load dataset from resources
      print("\nLoading dataset from resources...")
      let dataset = try loadMockDataset()
      print("Dataset loaded: \(dataset.totalDocuments) documents across \(dataset.categories.count) categories")
      for category in dataset.categories {
        print("   \(category.name): \(category.documents.count) docs")
      }

      // Setup database
      print("\nSetting up database...")
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: numResults,
        threshold: threshold,
        modelId: modelId
      )

      // Reset database
      print("\nResetting database...")
      try await db.reset()
      print("Database reset complete")

      // Add documents with timing
      print("\nIndexing documents...")
      let startTime = ContinuousClock.now
      let allTexts = dataset.allDocuments
      let ids = try await db.addDocuments(texts: allTexts)
      let indexDuration = ContinuousClock.now - startTime
      let docsPerSecond = Double(ids.count) / indexDuration.timeInterval

      print(
        "Indexed \(ids.count) documents in \(String(format: "%.2f", indexDuration.timeInterval))s " +
        "(\(String(format: "%.1f", docsPerSecond)) docs/sec)"
      )

      // Verify document count
      let docCount = try await db.documentCount
      print("Database contains \(docCount) documents")

      // Perform semantic search demonstrations
      try await performSearchDemonstrations(
        db: db,
        docCount: docCount,
        docsPerSecond: docsPerSecond
      )

      // Demonstrate update and delete operations
      print("\n" + "=" * 60)
      print("CRUD OPERATIONS DEMO")
      print("=" * 60)

      // Update a few documents
      print("\nUpdating 3 sample documents...")
      let updateIds = Array(ids.prefix(3))
      for (i, id) in updateIds.enumerated() {
        let newText = "Updated document \(i + 1): This content has been modified to test update functionality."
        try await db.updateDocument(id: id, newText: newText)
      }
      print("Updated \(updateIds.count) documents")

      // Delete some documents
      print("\nDeleting 5 sample documents...")
      let deleteIds = Array(ids.suffix(5))
      try await db.deleteDocuments(ids: deleteIds)
      print("Deleted \(deleteIds.count) documents")

      let finalCount = try await db.documentCount
      print("\nFinal document count: \(finalCount) (started with \(docCount))")

      print("\n" + "=" * 60)
      print("Mock Demonstration Complete!")
      print("=" * 60)
    }

    private func performSearchDemonstrations(
      db: VecturaKit,
      docCount: Int,
      docsPerSecond: Double
    ) async throws {
      print("\n" + "=" * 60)
      print("SEMANTIC SEARCH DEMONSTRATIONS")
      print("=" * 60)

      let searchQueries = [
        ("artificial intelligence", "Technology & ML concepts"),
        ("space exploration", "Astronomy & space-related topics"),
        ("leadership principles", "Management & business leadership"),
        ("wellness and fitness", "Health & wellbeing"),
        ("environmental conservation", "Climate & nature topics"),
        ("creative writing", "Literature & storytelling"),
        ("ancient civilizations", "Historical societies"),
        ("molecular biology", "Science & biology")
      ]

      var totalSearchTime: TimeInterval = 0

      for (index, (query, description)) in searchQueries.enumerated() {
        print("\n" + "-" * 60)
        print("Query \(index + 1): \"\(query)\"")
        print("Description: \(description)")
        print("-" * 60)

        let searchStart = ContinuousClock.now
        let results = try await db.search(
          query: .text(query),
          numResults: numResults,
          threshold: threshold
        )
        let searchDuration = ContinuousClock.now - searchStart
        totalSearchTime += searchDuration.timeInterval

        print("Search time: \(String(format: "%.1f", searchDuration.timeInterval * 1000))ms")
        print("Found \(results.count) results", terminator: "")

        if !results.isEmpty {
          let avgScore = results.map { $0.score }.reduce(0, +) / Float(results.count)
          let maxScore = results.map { $0.score }.max() ?? 0
          let minScore = results.map { $0.score }.min() ?? 0
          print(
            " (scores: \(String(format: "%.3f", minScore))-\(String(format: "%.3f", maxScore)), " +
            "avg: \(String(format: "%.3f", avgScore)))"
          )

          print("\nTop Results:")
          for (i, result) in results.prefix(3).enumerated() {
            let preview = result.text.prefix(80)
            print(
              "   \(i + 1). [\(String(format: "%.3f", result.score))] \(preview)" +
              "\(result.text.count > 80 ? "..." : "")"
            )
          }
        } else {
          print(" (below threshold)")
        }
      }

      print("\n" + "=" * 60)
      print("PERFORMANCE SUMMARY")
      print("=" * 60)
      let avgSearchTime = totalSearchTime / Double(searchQueries.count)
      print("Total documents indexed: \(docCount)")
      print("Indexing performance: \(String(format: "%.1f", docsPerSecond)) docs/sec")
      print("Total search queries: \(searchQueries.count)")
      print("Average search time: \(String(format: "%.1f", avgSearchTime * 1000))ms")
      print("Model: \(modelId)")
      print("Vector dimension: \(dimension)")
    }

    private func loadMockDataset() throws -> MockDataset {
      // .copy() in Package.swift flattens the directory structure
      guard let url = Bundle.module.url(forResource: "mock_documents", withExtension: "json") else {
        throw VecturaError.loadFailed("Could not find mock_documents.json in resources")
      }

      let data = try Data(contentsOf: url)
      let decoder = JSONDecoder()
      return try decoder.decode(MockDataset.self, from: data)
    }
  }

  struct Add: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Add documents to the vector database"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 384

    @Option(name: [.long, .customShort("m")], help: "Model ID for embeddings")
    var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"

    @Argument(help: "Text content to add")
    var text: [String]

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        modelId: modelId
      )
      let ids = try await db.addDocuments(texts: text)
      print("Added \(ids.count) documents:")
      for (id, text) in zip(ids, text) {
        print("ID: \(id)")
        print("Text: \(text)")
        print("---")
      }
    }
  }

  struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Search documents in the vector database"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 384

    @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold")
    var threshold: Float = 0.7

    @Option(name: [.long, .customShort("n")], help: "Number of results to return")
    var numResults: Int = 10

    @Option(name: [.long, .customShort("m")], help: "Model ID for embeddings")
    var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"

    @Argument(help: "Search query")
    var query: String

    mutating func run() async throws {
      guard !query.isEmpty else {
        VecturaCLI.writeError("Error: Query cannot be empty.")
        throw ExitCode.failure
      }

      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: numResults,
        threshold: threshold,
        modelId: modelId
      )
      let results = try await db.search(
        query: .text(query),
        numResults: numResults,
        threshold: threshold
      )

      print("Found \(results.count) results:")
      for result in results {
        print("ID: \(result.id)")
        print("Text: \(result.text)")
        print("Score: \(result.score)")
        print("Created: \(result.createdAt)")
        print("---")
      }
    }
  }

  struct Update: AsyncParsableCommand, Decodable {
    static let configuration = CommandConfiguration(
      abstract: "Update a document in the vector database"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 384

    @Option(name: [.long, .customShort("m")], help: "Model ID for embeddings")
    var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"

    @Argument(help: "Document ID to update")
    var id: DocumentID

    @Argument(help: "New text content")
    var newText: String

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        modelId: modelId
      )
      try await db.updateDocument(id: id.uuid, newText: newText)
      print("Updated document \(id.uuid) with new text: \(newText)")
    }
  }

  struct Delete: AsyncParsableCommand, Decodable {
    static let configuration = CommandConfiguration(
      abstract: "Delete documents from the vector database"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 384

    @Argument(help: "Document IDs to delete")
    var ids: [DocumentID]

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        modelId: "sentence-transformers/all-MiniLM-L6-v2"
      )
      try await db.deleteDocuments(ids: ids.map(\.uuid))
      print("Deleted \(ids.count) documents")
    }
  }

  struct Reset: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Reset the vector database"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: [.long, .customShort("v")], help: "Vector dimension")
    var dimension: Int = 384

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        modelId: "sentence-transformers/all-MiniLM-L6-v2"
      )
      try await db.reset()
      print("Database reset successfully")
    }
  }
}
