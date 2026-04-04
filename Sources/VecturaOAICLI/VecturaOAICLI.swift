import ArgumentParser
import Foundation
import VecturaKit
import VecturaOAIKit

@main
struct VecturaOAICLI: AsyncParsableCommand {
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

    struct ConnectionOptions: ParsableArguments {
        @Option(name: [.long], help: "OpenAI-compatible base URL, for example http://localhost:1234/v1")
        var baseURL: String = ProcessInfo.processInfo.environment["VECTURA_OAI_BASE_URL"] ?? "http://localhost:1234/v1"

        @Option(name: [.long], help: "Embedding model identifier")
        var model: String = ProcessInfo.processInfo.environment["VECTURA_OAI_MODEL"] ?? ""

        @Option(name: [.long], help: "Optional API key")
        var apiKey: String = ProcessInfo.processInfo.environment["VECTURA_OAI_API_KEY"] ?? ""

        @Option(name: [.long], help: "Request timeout in seconds")
        var timeout: Double = 120

        @Option(name: [.long], help: "HTTP 429 retry attempts")
        var retryAttempts: Int = 2

        @Option(name: [.long], help: "Base retry delay in seconds")
        var retryBaseDelay: Double = 1
    }

    static let configuration = CommandConfiguration(
        commandName: "vectura-oai",
        abstract: "A CLI tool for VecturaKit vector databases using OpenAI-compatible embeddings",
        subcommands: [Add.self, Search.self, Update.self, Delete.self, Reset.self, Mock.self]
    )

    static func writeError(_ message: String) {
        let errorMessage = message + "\n"
        if let data = errorMessage.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    static func setupDB(
        dbName: String,
        directory: String?,
        dimension: Int? = nil,
        numResults: Int = 10,
        threshold: Float = 0.7,
        connection: ConnectionOptions
    ) async throws -> VecturaKit {
        guard !connection.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Missing embedding model. Pass --model or set VECTURA_OAI_MODEL.")
        }

        let directoryURL = directory.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let config = try VecturaConfig(
            name: dbName,
            directoryURL: directoryURL,
            dimension: dimension,
            searchOptions: .init(
                defaultNumResults: numResults,
                minThreshold: threshold
            )
        )

        let embedder = OpenAICompatibleEmbedder(
            baseURL: connection.baseURL,
            model: connection.model,
            apiKey: connection.apiKey.isEmpty ? nil : connection.apiKey,
            timeoutInterval: connection.timeout,
            retryAttempts: connection.retryAttempts,
            retryBaseDelaySeconds: connection.retryBaseDelay
        )

        return try await VecturaKit(config: config, embedder: embedder)
    }
}

extension VecturaOAICLI {
    struct Mock: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Run a mock demonstration with sample data"
        )

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold")
        var threshold: Float = 0.7

        @Option(name: [.long, .customShort("n")], help: "Number of results to return")
        var numResults: Int = 10

        mutating func run() async throws {
            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                numResults: numResults,
                threshold: threshold,
                connection: connection
            )

            try await db.reset()

            let sampleTexts = [
                "The quick brown fox jumps over the lazy dog",
                "To be or not to be, that is the question",
                "All that glitters is not gold",
                "A journey of a thousand miles begins with a single step",
                "Where there is smoke, there is fire",
            ]

            let ids = try await db.addDocuments(texts: sampleTexts)
            print("Added \(ids.count) documents")

            let results = try await db.search(query: .text("journey"), numResults: numResults, threshold: threshold)
            print("Found \(results.count) results for 'journey'")
            for result in results {
                print("ID: \(result.id)")
                print("Text: \(result.text)")
                print("Score: \(result.score)")
                print("---")
            }
        }
    }

    struct Add: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Add documents to the vector database")

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        @Argument(help: "Text content to add")
        var text: [String]

        mutating func run() async throws {
            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                connection: connection
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
        static let configuration = CommandConfiguration(abstract: "Search documents in the vector database")

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold")
        var threshold: Float?

        @Option(name: [.long, .customShort("n")], help: "Number of results to return")
        var numResults: Int?

        @Argument(help: "Search query")
        var query: String

        mutating func run() async throws {
            guard !query.isEmpty else {
                VecturaOAICLI.writeError("Error: Query cannot be empty.")
                throw ExitCode.failure
            }

            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                numResults: numResults ?? 10,
                threshold: threshold ?? 0.7,
                connection: connection
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
        static let configuration = CommandConfiguration(abstract: "Update a document in the vector database")

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        @Argument(help: "Document ID to update")
        var id: DocumentID

        @Argument(help: "New text content")
        var newText: String

        mutating func run() async throws {
            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                connection: connection
            )
            try await db.updateDocument(id: id.uuid, newText: newText)
            print("Updated document \(id.uuid)")
        }
    }

    struct Delete: AsyncParsableCommand, Decodable {
        static let configuration = CommandConfiguration(abstract: "Delete documents from the vector database")

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        @Argument(help: "Document IDs to delete")
        var ids: [DocumentID]

        mutating func run() async throws {
            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                connection: connection
            )
            try await db.deleteDocuments(ids: ids.map(\.uuid))
            print("Deleted \(ids.count) documents")
        }
    }

    struct Reset: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Reset the vector database")

        @OptionGroup var connection: ConnectionOptions

        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-oai-cli-db"

        @Option(name: [.long], help: "Database directory")
        var directory: String?

        @Option(name: [.long, .customShort("v")], help: "Vector dimension (auto-detected if not specified)")
        var dimension: Int?

        mutating func run() async throws {
            let db = try await VecturaOAICLI.setupDB(
                dbName: dbName,
                directory: directory,
                dimension: dimension,
                connection: connection
            )
            try await db.reset()
            print("Database reset complete")
        }
    }
}
