import Foundation
import MLX
import MLXEmbedders

/// A protocol defining the requirements for a vector database instance.
@available(macOS 13.0, iOS 16.0, *)
public protocol VecturaProtocol {
    /// Adds a document to the vector store.
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - embedding: The vector embedding of the document.
    ///   - id: Optional unique identifier for the document.
    /// - Returns: The ID of the added document.
    func addDocument(
        text: String,
        embedding: MLXArray,
        id: UUID?
    ) async throws -> UUID
    
    /// Searches for similar documents using a query vector.
    /// - Parameters:
    ///   - query: The query vector to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by similarity.
    func search(
        query: MLXArray,
        numResults: Int?,
        threshold: Float?
    ) async throws -> [VecturaSearchResult]
    
    /// Removes all documents from the store.
    func reset() async throws
}

// End of file. No additional code.
