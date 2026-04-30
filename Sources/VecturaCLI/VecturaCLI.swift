import ArgumentParser
import Foundation
import NaturalLanguage
import VecturaKit
import VecturaNLKit

struct MockDataset {
  struct Category {
    let name: String
    let documents: [String]
  }

  let categories: [Category]

  var totalDocuments: Int {
    categories.reduce(0) { $0 + $1.documents.count }
  }

  var allDocuments: [String] {
    categories.flatMap(\.documents)
  }

  static let `default` = MockDataset(categories: [
    Category(name: "Technology", documents: [
      "Apple platforms rely on Swift for app development across iPhone, iPad, Mac, Apple TV, and Vision Pro.",
      "Semantic search combines vector similarity with lexical ranking to improve retrieval quality.",
      "Distributed systems use queues, caches, and retry strategies to keep latency predictable.",
      "Mobile teams measure p95 latency and crash-free sessions before shipping major releases.",
    ]),
    Category(name: "Science", documents: [
      "Astronomers analyze light from distant galaxies to understand how the universe expands.",
      "Molecular biology studies proteins, RNA, and DNA interactions inside living cells.",
      "Climate scientists model atmospheric warming, ocean circulation, and carbon emissions.",
      "Robotics blends control systems, perception, and planning for physical automation.",
    ]),
    Category(name: "Business", documents: [
      "Product managers align roadmap priorities with customer feedback and technical constraints.",
      "Leadership principles matter when teams need clear decisions during incidents.",
      "Healthy teams write down operating procedures before scaling support rotations.",
      "Finance organizations track cash flow, revenue quality, and cost efficiency over time.",
    ]),
    Category(name: "Creative", documents: [
      "Creative writing workshops encourage revision, feedback, and careful scene construction.",
      "Historical fiction often uses archival research to ground imagined dialogue in real events.",
      "Storytelling works better when conflict, pacing, and character motivation stay coherent.",
      "Music theory explains harmony, rhythm, melody, and tonal movement across styles.",
    ]),
  ])
}

private extension String {
  static func * (left: String, right: Int) -> String {
    String(repeating: left, count: right)
  }
}

private extension Duration {
  var timeInterval: TimeInterval {
    let (seconds, attoseconds) = components
    return TimeInterval(seconds) + (TimeInterval(attoseconds) / 1e18)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@main
struct VecturaCLI: AsyncParsableCommand {
  struct DocumentID: ExpressibleByArgument {
    let uuid: UUID

    init?(argument: String) {
      guard let uuid = UUID(uuidString: argument) else { return nil }
      self.uuid = uuid
    }
  }

  struct LanguageOption: ExpressibleByArgument {
    let language: NLLanguage

    init?(argument: String) {
      let normalized = argument.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      switch normalized {
      case "en", "english":
        language = .english
      case "es", "spanish":
        language = .spanish
      case "fr", "french":
        language = .french
      case "de", "german":
        language = .german
      case "it", "italian":
        language = .italian
      case "pt", "portuguese":
        language = .portuguese
      default:
        language = NLLanguage(rawValue: normalized)
      }
    }
  }

  static let configuration = CommandConfiguration(
    commandName: "vectura",
    abstract: "A CLI tool for VecturaKit using NaturalLanguage embeddings",
    subcommands: [Add.self, Search.self, Update.self, Delete.self, Reset.self, Mock.self]
  )

  static func writeError(_ message: String) {
    guard let data = (message + "\n").data(using: .utf8) else { return }
    FileHandle.standardError.write(data)
  }

  static func setupDB(
    dbName: String,
    directoryURL: URL?,
    dimension: Int?,
    numResults: Int,
    threshold: Float,
    language: NLLanguage
  ) async throws -> VecturaKit {
    let config = try VecturaConfig(
      name: dbName,
      directoryURL: directoryURL,
      dimension: dimension,
      searchOptions: VecturaConfig.SearchOptions(
        defaultNumResults: numResults,
        minThreshold: threshold
      )
    )
    let embedder = try await NLContextualEmbedder(language: language)
    return try await VecturaKit(config: config, embedder: embedder)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension VecturaCLI {
  struct Mock: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Run a local demo with NaturalLanguage embeddings"
    )

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-demo-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold for searches")
    var threshold: Float = 0.5

    @Option(name: [.long, .customShort("n")], help: "Number of results to return per search")
    var numResults: Int = 5

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    mutating func run() async throws {
      print("VecturaKit Mock Demonstration")
      print("=" * 60)

      let dataset = MockDataset.default
      print("\nLoaded \(dataset.totalDocuments) documents across \(dataset.categories.count) categories")
      for category in dataset.categories {
        print("  \(category.name): \(category.documents.count)")
      }

      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: numResults,
        threshold: threshold,
        language: language.language
      )

      try await db.reset()

      print("\nIndexing documents...")
      let start = ContinuousClock.now
      let ids = try await db.addDocuments(texts: dataset.allDocuments)
      let duration = ContinuousClock.now - start
      let docsPerSecond = Double(ids.count) / max(duration.timeInterval, 0.001)
      print(
        "Indexed \(ids.count) documents in \(String(format: "%.2f", duration.timeInterval))s "
          + "(\(String(format: "%.1f", docsPerSecond)) docs/sec)"
      )

      let queries = [
        "swift app development",
        "leadership decisions",
        "molecular biology",
        "creative storytelling",
      ]

      for query in queries {
        print("\n" + "-" * 60)
        print("Query: \(query)")
        let results = try await db.search(
          query: .text(query),
          numResults: numResults,
          threshold: threshold
        )
        if results.isEmpty {
          print("No results")
          continue
        }

        for (index, result) in results.enumerated() {
          let preview = String(result.text.prefix(90))
          print("\(index + 1). [\(String(format: "%.3f", result.score))] \(preview)")
        }
      }

      print("\nFinal document count: \(try await db.documentCount)")
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }

  struct Add: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Add documents to the vector database")

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    @Argument(help: "Text content to add")
    var text: [String]

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        language: language.language
      )

      let ids = try await db.addDocuments(texts: text)
      for (id, value) in zip(ids, text) {
        print("ID: \(id)")
        print("Text: \(value)")
        print("---")
      }
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }

  struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Search documents in the vector database")

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold")
    var threshold: Float = 0.7

    @Option(name: [.long, .customShort("n")], help: "Number of results to return")
    var numResults: Int = 10

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    @Argument(help: "Search query")
    var query: String

    mutating func run() async throws {
      guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        VecturaCLI.writeError("Error: Query cannot be empty.")
        throw ExitCode.failure
      }

      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: numResults,
        threshold: threshold,
        language: language.language
      )
      let results = try await db.search(
        query: .text(query),
        numResults: numResults,
        threshold: threshold
      )

      for result in results {
        print("ID: \(result.id)")
        print("Text: \(result.text)")
        print("Score: \(result.score)")
        print("Created: \(result.createdAt)")
        print("---")
      }
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }

  struct Update: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Update a document in the vector database")

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    @Argument(help: "Document ID to update")
    var id: DocumentID

    @Argument(help: "New text content")
    var newText: String

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        language: language.language
      )
      try await db.updateDocument(id: id.uuid, newText: newText)
      print("Updated document \(id.uuid)")
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }

  struct Delete: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Delete documents from the vector database")

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    @Argument(help: "Document IDs to delete")
    var ids: [DocumentID]

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        language: language.language
      )
      try await db.deleteDocuments(ids: ids.map(\.uuid))
      print("Deleted \(ids.count) documents")
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }

  struct Reset: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Reset the vector database")

    @Option(name: [.long, .customShort("d")], help: "Database name")
    var dbName: String = "vectura-cli-db"

    @Option(name: .long, help: "Optional database directory")
    var directory: String?

    @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if omitted)")
    var dimension: Int?

    @Option(name: [.long, .customShort("l")], help: "Embedding language, for example en or english")
    var language: LanguageOption = .init(language: .english)

    mutating func run() async throws {
      let db = try await VecturaCLI.setupDB(
        dbName: dbName,
        directoryURL: directoryURL,
        dimension: dimension,
        numResults: 10,
        threshold: 0.7,
        language: language.language
      )
      try await db.reset()
      print("Database reset successfully")
    }

    private var directoryURL: URL? {
      directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
  }
}

private extension VecturaCLI.LanguageOption {
  init(language: NLLanguage) {
    self.language = language
  }
}
