import ArgumentParser
import Foundation
import MLXEmbedders
import VecturaKit
import VecturaMLXKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
@main
struct VecturaMLXCLI: AsyncParsableCommand {
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
        commandName: "vectura-mlx",
        abstract: "A CLI tool for VecturaMLXKit vector database using MLX",
        subcommands: [Add.self, Search.self, Update.self, Delete.self, Reset.self, Mock.self]
    )
    
    static func setupDB(
        dbName: String, modelConfiguration: MLXEmbedders.ModelConfiguration = .nomic_text_v1_5
    )
    async throws
    -> VecturaMLXKit
    {
        let config = VecturaConfig(
            name: dbName,
            dimension: 768  // nomic_text_v1_5 model outputs 768-dimensional embeddings
        )
        return try await VecturaMLXKit(config: config, modelConfiguration: modelConfiguration)
    }
}

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
extension VecturaMLXCLI {
    struct Mock: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Run a mock demonstration with sample data"
        )
        
        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-mlx-cli-db"
        
        mutating func run() async throws {
            print("Starting mock command...")
            
            print("Setting up database...")
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
            print("Database setup complete")
            
            // First, reset the database
            print("\n🧹 Resetting database...")
            try await db.reset()
            print("Reset complete")
            
            // Add sample documents
            print("\n📝 Adding sample documents...")
            let sampleTexts = [
                "The quick brown fox jumps over the lazy dog",
                "To be or not to be, that is the question",
                "All that glitters is not gold",
                "A journey of a thousand miles begins with a single step",
                "Where there's smoke, there's fire",
            ]
            
            let ids = try await db.addDocuments(texts: sampleTexts)
            print("Added \(ids.count) documents:")
            for (id, text) in zip(ids, sampleTexts) {
                print("ID: \(id)")
                print("Text: \(text)")
                print("---")
            }
            
            // Search for documents
            print("\n🔍 Searching for 'journey'...")
            let results = try await db.search(query: "journey")
            
            print("Found \(results.count) results:")
            for result in results {
                print("ID: \(result.id)")
                print("Text: \(result.text)")
                print("Score: \(result.score)")
                print("Created: \(result.createdAt)")
                print("---")
            }
            
            // Update a document
            if let firstId = ids.first {
                print("\n✏️ Updating first document...")
                let newText = "The quick red fox jumps over the sleeping dog"
                try await db.updateDocument(id: firstId, newText: newText)
                print("Updated document \(firstId) with new text: \(newText)")
            }
            
            // Delete last document
            if let lastId = ids.last {
                print("\n🗑️ Deleting last document...")
                try await db.deleteDocuments(ids: [lastId])
                print("Deleted document \(lastId)")
            }
            
            print("\n✨ Mock demonstration completed!")
        }
    }
    
    struct Add: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Add documents to the vector database"
        )
        
        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-mlx-cli-db"
        
        @Argument(help: "Text content to add")
        var text: [String]
        
        mutating func run() async throws {
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
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
        var dbName: String = "vectura-mlx-cli-db"
        
        @Option(name: [.long, .customShort("t")], help: "Minimum similarity threshold")
        var threshold: Float?
        
        @Option(name: [.long, .customShort("n")], help: "Number of results to return")
        var numResults: Int?
        
        @Argument(help: "Search query")
        var query: String
        
        mutating func run() async throws {
            guard !query.isEmpty else {
                print("Error: Query cannot be empty.")
                throw ExitCode.failure
            }
            
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
            let results = try await db.search(
                query: query,
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
        var dbName: String = "vectura-mlx-cli-db"
        
        @Argument(help: "Document ID to update")
        var id: DocumentID
        
        @Argument(help: "New text content")
        var newText: String
        
        mutating func run() async throws {
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
            try await db.updateDocument(id: id.uuid, newText: newText)
            print("Updated document \(id.uuid) with new text: \(newText)")
        }
    }
    
    struct Delete: AsyncParsableCommand, Decodable {
        static let configuration = CommandConfiguration(
            abstract: "Delete documents from the vector database"
        )
        
        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-mlx-cli-db"
        
        @Argument(help: "Document IDs to delete")
        var ids: [DocumentID]
        
        mutating func run() async throws {
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
            try await db.deleteDocuments(ids: ids.map(\.uuid))
            print("Deleted \(ids.count) documents")
        }
    }
    
    struct Reset: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Reset the vector database"
        )
        
        @Option(name: [.long, .customShort("d")], help: "Database name")
        var dbName: String = "vectura-mlx-cli-db"
        
        mutating func run() async throws {
            let db = try await VecturaMLXCLI.setupDB(dbName: dbName)
            try await db.reset()
            print("Database reset successfully")
        }
    }
}
