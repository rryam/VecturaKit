import Foundation

/// Search query types supported by search engines
public enum SearchQuery: Sendable {
    /// Text-based query
    case text(String)

    /// Vector-based query
    case vector([Float])

    /// Hybrid query with both text and precomputed vector
    case hybrid(text: String, vector: [Float])
}

/// Search options for configuring search behavior
public struct SearchOptions: Sendable {
    /// Maximum number of results to return
    public var numResults: Int

    /// Minimum similarity threshold (0.0-1.0)
    public var threshold: Float?

    public init(
        numResults: Int = 10,
        threshold: Float? = nil
    ) {
        self.numResults = numResults
        self.threshold = threshold
    }
}

/// Context provided to search engines for accessing documents
public protocol SearchContext: Sendable {
    /// Get all documents (used by full-memory search)
    func getAllDocuments() async throws -> [VecturaDocument]

    /// Get specific documents by IDs (used by indexed search)
    func getDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument]

    /// Get total document count
    func getDocumentCount() async throws -> Int

    /// Perform storage-layer search if available (returns candidate IDs)
    func storageSearch(vector: [Float], topK: Int) async throws -> [UUID]?
}

/// Protocol for search engine implementations
public protocol VecturaSearchEngine: Sendable {
    /// Execute search with the given query
    /// - Parameters:
    ///   - query: The search query (text, vector, or hybrid)
    ///   - context: Context for accessing documents
    ///   - options: Search configuration options
    /// - Returns: Array of search results sorted by relevance
    func search(
        query: SearchQuery,
        context: SearchContext,
        options: SearchOptions
    ) async throws -> [VecturaSearchResult]

    /// Index a document (optional, some engines don't need this)
    func indexDocument(_ document: VecturaDocument) async throws

    /// Remove a document from index (optional)
    func removeDocument(id: UUID) async throws
}
