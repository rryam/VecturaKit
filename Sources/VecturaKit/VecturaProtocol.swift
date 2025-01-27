import Foundation

/// A protocol defining the requirements for a vector database instance.
public protocol VecturaProtocol {

    /// Adds multiple documents to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - ids: Optional unique identifiers for the documents.
    ///   - model: A ``VecturaModelSource`` specifying how to load the model.
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The IDs of the added documents.
    func addDocuments(
        texts: [String],
        ids: [UUID]?,
        model: VecturaModelSource
    ) async throws -> [UUID]

    /// Searches for similar documents using a *pre-computed query embedding*.
    ///
    /// - Parameters:
    ///   - query: The query vector to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by similarity.
    func search(
        query: [Float],
        numResults: Int?,
        threshold: Float?
    ) async throws -> [VecturaSearchResult]

    /// Removes all documents from the vector store.
    func reset() async throws
}

// MARK: - Default Implementations

public extension VecturaProtocol {

    /// Adds a document to the vector store by embedding text.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    ///   - model: A ``VecturaModelSource`` specifying how to load the model.
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The ID of the added document.
    func addDocument(
        text: String,
        id: UUID? = nil,
        model: VecturaModelSource = .default
    ) async throws -> UUID {
        let ids = try await addDocuments(
            texts: [text],
            ids: id.map { [$0] },
            model: model
        )
        return ids[0]
    }

    /// Adds a document to the vector store by embedding text.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    ///   - modelId: Identifier of the model to use for generating the embedding
    ///              (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    /// - Returns: The ID of the added document.
    @_disfavoredOverload
    func addDocument(
        text: String,
        id: UUID?,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws -> UUID {
        try await addDocument(text: text, id: id, model: .id(modelId))
    }

    /// Adds multiple documents to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - ids: Optional unique identifiers for the documents.
    ///   - modelId: Identifier of the model to use for generating the embedding
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The IDs of the added documents.
    func addDocuments(
        texts: [String],
        ids: [UUID]? = nil,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws -> [UUID] {
        try await addDocuments(texts: texts, ids: ids, model: .id(modelId))
    }
}
