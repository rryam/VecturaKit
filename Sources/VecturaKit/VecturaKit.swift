import Foundation
import MLX
import MLXEmbedders

/// A vector database implementation that stores and searches documents using their vector embeddings.
public actor VecturaKit: VecturaProtocol {
    /// The configuration for this vector database instance.
    private let config: VecturaConfig
    
    /// The storage for documents.
    private var documents: [UUID: VecturaDocument]
    
    /// Creates a new vector database instance.
    /// - Parameter config: The configuration for the database.
    public init(config: VecturaConfig) {
        self.config = config
        self.documents = [:]
    }
    
    /// Adds a document to the vector store.
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
        
        documents[document.id] = document
        return document.id
    }
    
    /// Searches for similar documents using a query vector.
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
        
        // Normalize query vector
        let queryNorm = sqrt(sum(query * query))
        let normalizedQuery = query / queryNorm
        
        var results: [VecturaSearchResult] = []
        
        for document in documents.values {
            let vector = document.embedding
            let vectorNorm = sqrt(sum(vector * vector))
            let normalizedVector = vector / vectorNorm
            
            // Compute cosine similarity
            let similarity = sum(normalizedQuery * normalizedVector)
                .asArray(Float.self)[0]
            
            if let minThreshold = threshold ?? config.searchOptions.minThreshold,
               similarity < minThreshold {
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
        
        // Sort by similarity score and limit results
        results.sort { $0.score > $1.score }
        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
    }
    
    /// Removes all documents from the store.
    public func reset() async throws {
        documents.removeAll()
    }
}

// End of file. No additional code.
