import Foundation
import MLX
import MLXEmbedders

/// A vector database implementation that stores and searches documents using their vector embeddings.
public class VecturaKit: VecturaProtocol {

    /// The configuration for this vector database instance.
    private let config: VecturaConfig

    /// The storage for documents.
    private var documents: [UUID: VecturaDocument]

    /// The storage directory for documents.
    private let storageDirectory: URL

    /// Cached normalized embeddings for faster search
    private var normalizedEmbeddings: [UUID: MLXArray] = [:]

    /// Creates a new vector database instance.
    ///
    /// - Parameter config: The configuration for the database.
    public init(config: VecturaConfig) throws {
        self.config = config
        self.documents = [:]

        /// Create storage directory in the app's Documents directory
        self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("VecturaKit")
            .appendingPathComponent(config.name)

        /// Try to create directory and load existing documents
        try FileManager.default.createDirectory(
            at: storageDirectory,
            withIntermediateDirectories: true
        )

        try loadDocuments()
    }

    /// Adds a document to the vector store.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - embedding: The vector embedding of the document.
    ///   - id: Optional unique identifier for the document.
    /// - Returns: The ID of the added document.
    public func addDocument(
        text: String,
        embedding: MLXArray,
        id: UUID? = nil
    ) async throws -> UUID {
        guard embedding.shape.last == config.dimension else {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: embedding.shape.last ?? 0
            )
        }

        let document = VecturaDocument(
            id: id,
            text: text,
            embedding: embedding
        )

        // Pre-compute normalized embedding
        let norm = sqrt(sum(embedding * embedding))
        normalizedEmbeddings[document.id] = embedding / norm

        documents[document.id] = document
        try await saveDocument(document)
        return document.id
    }

    /// Searches for similar documents using a query vector.
    ///
    /// - Parameters:
    ///   - query: The query vector to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by similarity.
    public func search(
        query: MLXArray,
        numResults: Int? = nil,
        threshold: Float? = nil
    ) async throws -> [VecturaSearchResult] {
        guard query.shape.last == config.dimension else {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: query.shape.last ?? 0
            )
        }

        let queryNorm = sqrt(sum(query * query))
        let normalizedQuery = query / queryNorm

        var results: [VecturaSearchResult] = []

        for document in documents.values {
            let normalizedVector = normalizedEmbeddings[document.id]!
            let similarity = sum(normalizedQuery * normalizedVector)
                .asArray(Float.self)[0]

            if let minThreshold = threshold ?? config.searchOptions.minThreshold,
               similarity < minThreshold
            {
                continue
            }

            results.append(
                VecturaSearchResult(
                    id: document.id,
                    text: document.text,
                    score: similarity,
                    createdAt: document.createdAt
                )
            )
        }

        results.sort { $0.score > $1.score }
        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
    }

    /// Resets the vector database.
    public func reset() async throws {
        documents.removeAll()
        normalizedEmbeddings.removeAll()

        /// Remove all files from storage
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory,
            includingPropertiesForKeys: nil
        )

        for fileURL in fileURLs {
            try FileManager.default.removeItem(at: fileURL)
        }
    }

    /// Saves a document to storage.
    private func saveDocument(_ document: VecturaDocument) async throws {
        let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(document)
        try data.write(to: documentURL)
    }

    /// Loads existing documents from storage.
    private func loadDocuments() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory,
            includingPropertiesForKeys: nil
        )

        let decoder = JSONDecoder()
        var loadErrors: [String] = []

        for fileURL in fileURLs where fileURL.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let document = try decoder.decode(VecturaDocument.self, from: data)
                documents[document.id] = document
            } catch {
                loadErrors.append("Failed to load document at \(fileURL): \(error.localizedDescription)")
            }
        }

        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
        }
    }
}
