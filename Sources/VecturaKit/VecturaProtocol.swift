import Foundation

/// A protocol defining the requirements for a vector database instance.
public protocol VecturaProtocol {

    /// Adds a document to the vector store by embedding text.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    ///   - modelId: Identifier of the model to use for generating the embedding
    ///              (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    /// - Returns: The ID of the added document.
    func addDocument(
        text: String,
        id: UUID?,
        modelId: String
    ) async throws -> UUID

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
