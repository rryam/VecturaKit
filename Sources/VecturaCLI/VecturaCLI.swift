import ArgumentParser
import Foundation
import VecturaKit

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
    
    static func setupDB(dbName: String, dimension: Int, numResults: Int, threshold: Float) async throws
    -> VecturaKit
    {
        let config = try VecturaConfig(
            name: dbName,
            dimension: dimension,
            searchOptions: VecturaConfig.SearchOptions(
                defaultNumResults: numResults,
                minThreshold: threshold
            )
        )
        return try await VecturaKit(config: config)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension VecturaCLI {
    struct Mock: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Run a mock demonstration with sample data"
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
        
        mutating func run() async throws {
            let db = try await VecturaCLI.setupDB(
                dbName: dbName,
                dimension: dimension,
                numResults: numResults,
                threshold: threshold
            )
            
            // First, reset the database
            print("\n🧹 Resetting database...")
            try await db.reset()
            
            // Add sample documents
            print("\n📝 Adding sample documents...")
            let sampleTexts = [
                "The quick brown fox jumps over the lazy dog",
                "To be or not to be, that is the question",
                "All that glitters is not gold",
                "A journey of a thousand miles begins with a single step",
                "Where there's smoke, there's fire",
            ]
            
            let ids = try await db.addDocuments(texts: sampleTexts, modelId: modelId)
            print("Added \(ids.count) documents:")
            for (id, text) in zip(ids, sampleTexts) {
                print("ID: \(id)")
                print("Text: \(text)")
                print("---")
            }
            
            // Search for documents
            print("\n🔍 Searching for 'journey'...")
            let results = try await db.search(
                query: "journey",
                numResults: numResults,
                threshold: threshold,
                modelId: modelId
            )
            
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
                try await db.updateDocument(id: firstId, newText: newText, modelId: modelId)
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
                threshold: 0.7
            )
            let ids = try await db.addDocuments(texts: text, modelId: modelId)
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
            let db = try await VecturaCLI.setupDB(
                dbName: dbName,
                dimension: dimension,
                numResults: numResults,
                threshold: threshold
            )
            let results = try await db.search(
                query: query,
                numResults: numResults,
                threshold: threshold,
                modelId: modelId
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
                threshold: 0.7
            )
            try await db.updateDocument(id: id.uuid, newText: newText, modelId: modelId)
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
                threshold: 0.7
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
                threshold: 0.7
            )
            try await db.reset()
            print("Database reset successfully")
        }
    }
}
