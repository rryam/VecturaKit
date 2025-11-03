import Accelerate
import Foundation

/// Vector search engine using VecturaKit's Accelerate-based similarity computation
public struct VectorSearchEngine: VecturaSearchEngine {
    /// The embedder for generating query vectors
    public let embedder: VecturaEmbedder

    /// Search strategy configuration
    public let strategy: Strategy

    /// Memory strategy for vector search
    public enum Strategy: Sendable {
        /// Use in-memory search (load all documents)
        case inMemory

        /// Use indexed search with two-stage filtering
        case indexed(candidateMultiplier: Int)

        /// Automatically choose based on document count
        case automatic(threshold: Int)
    }

    public init(
        embedder: VecturaEmbedder,
        strategy: Strategy = .automatic(threshold: 10_000)
    ) {
        self.embedder = embedder
        self.strategy = strategy
    }

    // MARK: - VecturaSearchEngine Protocol

    public func search(
        query: SearchQuery,
        context: SearchContext,
        options: SearchOptions
    ) async throws -> [VecturaSearchResult] {
        // Extract query vector
        let queryVector: [Float]
        switch query {
        case .vector(let vec):
            queryVector = vec
        case .text(let text):
            queryVector = try await embedder.embed(text: text)
        case .hybrid(_, let vec):
            queryVector = vec
        }

        // Choose search strategy
        let shouldUseIndexed = try await shouldUseIndexedSearch(context: context)

        if shouldUseIndexed {
            let multiplier = getMultiplier()
            return try await searchWithIndex(
                queryVector: queryVector,
                context: context,
                options: options,
                candidateMultiplier: multiplier
            )
        } else {
            return try await searchInMemory(
                queryVector: queryVector,
                context: context,
                options: options
            )
        }
    }

    public func indexDocument(_ document: VecturaDocument) async throws {
        // Vector search doesn't need additional indexing
    }

    public func removeDocument(id: UUID) async throws {
        // Vector search doesn't need additional cleanup
    }

    // MARK: - Search Implementations

    private func searchInMemory(
        queryVector: [Float],
        context: SearchContext,
        options: SearchOptions
    ) async throws -> [VecturaSearchResult] {
        let documents = try await context.getAllDocuments()

        guard !documents.isEmpty else {
            return []
        }

        let dimension = queryVector.count

        // Normalize query vector
        let normalizedQuery = try normalizeEmbedding(queryVector)

        // Build matrix of normalized document embeddings
        var docIds = [UUID]()
        var matrix = [Float]()
        matrix.reserveCapacity(documents.count * dimension)

        for doc in documents {
            let normalized = try normalizeEmbedding(doc.embedding)
            docIds.append(doc.id)
            matrix.append(contentsOf: normalized)
        }

        let docsCount = docIds.count
        var similarities = [Float](repeating: 0, count: docsCount)

        // Compute similarities using matrix-vector multiplication
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            docsCount,
            dimension,
            1.0,
            matrix,
            dimension,
            normalizedQuery,
            1,
            0.0,
            &similarities,
            1
        )

        // Build results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(docsCount)

        for (i, similarity) in similarities.enumerated() {
            if let threshold = options.threshold, similarity < threshold {
                continue
            }

            let doc = documents.first { $0.id == docIds[i] }!
            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: similarity,
                    createdAt: doc.createdAt
                )
            )
        }

        results.sort { $0.score > $1.score }
        return Array(results.prefix(options.numResults))
    }

    private func searchWithIndex(
        queryVector: [Float],
        context: SearchContext,
        options: SearchOptions,
        candidateMultiplier: Int
    ) async throws -> [VecturaSearchResult] {
        let prefilterSize = options.numResults * candidateMultiplier

        // Try storage-layer search first
        let candidateIds: [UUID]
        if let storageIds = try await context.storageSearch(
            vector: queryVector,
            topK: prefilterSize
        ) {
            candidateIds = storageIds
        } else {
            // Fallback: get all documents and do in-memory prefilter
            _ = try await context.getAllDocuments()  // Ensure documents loaded
            let prefilterResults = try await searchInMemory(
                queryVector: queryVector,
                context: context,
                options: SearchOptions(numResults: prefilterSize)
            )
            candidateIds = prefilterResults.map { $0.id }
        }

        guard !candidateIds.isEmpty else {
            return []
        }

        // Load candidate documents
        let candidates = try await context.getDocuments(ids: candidateIds)

        guard !candidates.isEmpty else {
            return []
        }

        // Compute exact similarities
        let normalizedQuery = try normalizeEmbedding(queryVector)
        let dimension = queryVector.count

        var candidateDocIds = [UUID]()
        var candidateDocs = [VecturaDocument]()
        var matrix = [Float]()
        matrix.reserveCapacity(candidates.count * dimension)

        for (id, doc) in candidates {
            let normalized = try normalizeEmbedding(doc.embedding)
            candidateDocIds.append(id)
            candidateDocs.append(doc)
            matrix.append(contentsOf: normalized)
        }

        let candidatesCount = candidateDocIds.count
        var similarities = [Float](repeating: 0, count: candidatesCount)

        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            candidatesCount,
            dimension,
            1.0,
            matrix,
            dimension,
            normalizedQuery,
            1,
            0.0,
            &similarities,
            1
        )

        // Build results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(candidatesCount)

        for (i, similarity) in similarities.enumerated() {
            if let threshold = options.threshold, similarity < threshold {
                continue
            }

            let doc = candidateDocs[i]
            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: similarity,
                    createdAt: doc.createdAt
                )
            )
        }

        results.sort { $0.score > $1.score }
        return Array(results.prefix(options.numResults))
    }

    // MARK: - Helper Methods

    private func shouldUseIndexedSearch(context: SearchContext) async throws -> Bool {
        switch strategy {
        case .inMemory:
            return false
        case .indexed:
            return true
        case .automatic(let threshold):
            let count = try await context.getDocumentCount()
            return count >= threshold
        }
    }

    private func getMultiplier() -> Int {
        switch strategy {
        case .indexed(let multiplier):
            return multiplier
        case .automatic, .inMemory:
            return 10
        }
    }

    private func normalizeEmbedding(_ embedding: [Float]) throws -> [Float] {
        let norm = l2Norm(embedding)

        guard norm > 1e-10 else {
            throw VecturaError.invalidInput("Cannot normalize zero-norm embedding vector")
        }

        var divisor = norm
        var normalized = [Float](repeating: 0, count: embedding.count)
        vDSP_vsdiv(embedding, 1, &divisor, &normalized, 1, vDSP_Length(embedding.count))
        return normalized
    }

    private func l2Norm(_ v: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
        return sqrt(sumSquares)
    }
}
