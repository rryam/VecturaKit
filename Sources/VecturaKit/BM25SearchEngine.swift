import Foundation

/// BM25 text search engine
public actor BM25SearchEngine: VecturaSearchEngine {
    private var index: BM25Index?
    private let k1: Float
    private let b: Float

    public init(k1: Float = 1.2, b: Float = 0.75) {
        self.k1 = k1
        self.b = b
    }

    // MARK: - VecturaSearchEngine Protocol

    public func search(
        query: SearchQuery,
        context: SearchContext,
        options: SearchOptions
    ) async throws -> [VecturaSearchResult] {
        guard case .text(let queryText) = query else {
            throw VecturaError.invalidInput("BM25 only supports text queries")
        }

        // Ensure index is built
        if index == nil {
            let documents = try await context.getAllDocuments()
            index = BM25Index(documents: documents, k1: k1, b: b)
        }

        guard let index = index else {
            return []
        }

        let results = index.search(query: queryText, topK: options.numResults)

        // Filter by threshold
        var filteredResults = results
        if let threshold = options.threshold {
            filteredResults = results.filter { $0.score >= threshold }
        }

        return filteredResults.map { result in
            VecturaSearchResult(
                id: result.document.id,
                text: result.document.text,
                score: result.score,
                createdAt: result.document.createdAt
            )
        }
    }

    public func indexDocument(_ document: VecturaDocument) async throws {
        if var idx = index {
            idx.addDocument(document)
            index = idx
        }
    }

    public func removeDocument(id: UUID) async throws {
        if var idx = index {
            idx.removeDocument(id)
            index = idx
        }
    }
}
