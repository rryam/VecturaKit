import Foundation
@testable import VecturaKit

/// Mock implementation of IndexedVecturaStorage for performance testing.
///
/// This implementation provides realistic indexed storage behavior by:
/// - Storing documents in memory (simulating a database)
/// - Implementing approximate nearest neighbor search for `searchCandidates()`
/// - Supporting pagination and batch loading
///
/// Unlike FileStorageProvider, this mock supports true indexed mode testing.
actor MockIndexedStorage: IndexedVecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]
    private var documentList: [VecturaDocument] = []  // For stable ordering

    // MARK: - VecturaStorage Protocol

    func createStorageDirectoryIfNeeded() async throws {
        // No-op for in-memory storage
    }

    func loadDocuments() async throws -> [VecturaDocument] {
        return documentList
    }

    func saveDocument(_ document: VecturaDocument) async throws {
        if let index = documentList.firstIndex(where: { $0.id == document.id }) {
            documentList[index] = document
        } else {
            documentList.append(document)
        }
        documents[document.id] = document
    }

    func deleteDocument(withID id: UUID) async throws {
        documents[id] = nil
        documentList.removeAll { $0.id == id }
    }

    func updateDocument(_ document: VecturaDocument) async throws {
        if let index = documentList.firstIndex(where: { $0.id == document.id }) {
            documentList[index] = document
        }
        documents[document.id] = document
    }

    // MARK: - IndexedVecturaStorage Protocol

    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument] {
        let start = min(offset, documentList.count)
        let end = min(offset + limit, documentList.count)
        guard start < end else { return [] }
        return Array(documentList[start..<end])
    }

    func getTotalDocumentCount() async throws -> Int {
        return documentList.count
    }

    /// Performs approximate nearest neighbor search using cosine similarity.
    ///
    /// This implementation:
    /// 1. Computes cosine similarity for all documents
    /// 2. Returns top `prefilterSize` candidates (simulating ANN index)
    /// 3. Allows VecturaKit to perform exact rescoring on candidates
    func searchCandidates(
        queryEmbedding: [Float],
        topK: Int,
        prefilterSize: Int
    ) async throws -> [UUID] {
        // Calculate similarity scores for all documents
        var scores: [(id: UUID, score: Double)] = []

        for doc in documentList {
            let similarity = cosineSimilarity(queryEmbedding, doc.embedding)
            scores.append((id: doc.id, score: similarity))
        }

        // Sort by similarity (descending) and take top prefilterSize
        scores.sort { $0.score > $1.score }

        let candidateCount = min(prefilterSize, scores.count)
        return Array(scores.prefix(candidateCount).map { $0.id })
    }

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
        var result: [UUID: VecturaDocument] = [:]
        for id in ids {
            if let doc = documents[id] {
                result[id] = doc
            }
        }
        return result
    }

    // MARK: - Helper Methods

    /// Computes cosine similarity between two vectors.
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0.0 }

        var dotProduct: Double = 0.0
        var normA: Double = 0.0
        var normB: Double = 0.0

        for i in 0..<a.count {
            let aVal = Double(a[i])
            let bVal = Double(b[i])
            dotProduct += aVal * bVal
            normA += aVal * aVal
            normB += bVal * bVal
        }

        guard normA > 0, normB > 0 else { return 0.0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}
