//
//  BM25Index.swift
//  VecturaKit
//
//  Created by Rudrank Riyam on 1/19/25.
//

import Foundation

private func tokenize(_ text: String) -> [String] {
    text.lowercased()
        .folding(options: .diacriticInsensitive, locale: .current)
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

/// An index for BM25-based text search over VecturaDocuments
public struct BM25Index {
    private let k1: Float
    private let b: Float
    private var documents: [VecturaDocument]
    private var documentFrequencies: [String: Int]
    private var documentLengths: [UUID: Int]
    private var averageDocumentLength: Float

    /// Creates a new BM25 index for the given documents
    /// 
    /// - Parameters:
    ///   - documents: The documents to index
    ///   - k1: BM25 k1 parameter (default: 1.2)
    ///   - b: BM25 b parameter (default: 0.75)
    public init(documents: [VecturaDocument], k1: Float = 1.2, b: Float = 0.75) {
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.documentFrequencies = [:]

        self.documentLengths = documents.reduce(into: [:]) { dict, doc in
            dict[doc.id] = tokenize(doc.text).count
        }

        self.averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(documents.count)

        for document in documents {
            let terms = Set(tokenize(document.text))
            for term in terms {
                documentFrequencies[term, default: 0] += 1
            }
        }
    }

    /// Searches the index using BM25 scoring
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - topK: Maximum number of results to return
    /// - Returns: Array of tuples containing documents and their BM25 scores
    public func search(query: String, topK: Int = 10) -> [(document: VecturaDocument, score: Float)] {
        let queryTerms = tokenize(query)
        var scores: [(VecturaDocument, Float)] = []

        for document in documents {
            let docLength = Float(documentLengths[document.id] ?? 0)
            var score: Float = 0.0

            for term in queryTerms {
                let tf = termFrequency(term: term, in: document)
                let df = Float(documentFrequencies[term] ?? 0)

                let idf = log((Float(documents.count) - df + 0.5) / (df + 0.5))
                let numerator = tf * (k1 + 1)
                let denominator = tf + k1 * (1 - b + b * docLength / averageDocumentLength)

                score += idf * (numerator / denominator)
            }

            scores.append((document, score))
        }

        return scores
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .filter { $0.1 > 0 }
    }

    /// Add a new document to the index incrementally
    ///
    /// - Parameter document: The document to add
    public mutating func addDocument(_ document: VecturaDocument) {
        documents.append(document)

        let length = tokenize(document.text).count
        documentLengths[document.id] = length

        incrementTermFrequencies(for: document)

        updateAverageDocumentLength()
    }

    /// Remove a document from the index incrementally
    ///
    /// - Parameter documentID: The ID of the document to remove
    public mutating func removeDocument(_ documentID: UUID) {
        guard let index = documents.firstIndex(where: { $0.id == documentID }) else {
            return
        }

        let document = documents[index]
        documents.remove(at: index)

        decrementTermFrequencies(for: document)

        documentLengths.removeValue(forKey: documentID)
        updateAverageDocumentLength()
    }

    /// Update an existing document in the index incrementally
    ///
    /// - Parameter document: The updated document
    public mutating func updateDocument(_ document: VecturaDocument) {
        // If an old document with the same ID exists, remove its contribution to the index first.
        guard let oldDocIndex = documents.firstIndex(where: { $0.id == document.id }) else {
            // If document doesn't exist, this is an add operation.
            addDocument(document)
            return
        }
        let oldDocument = documents[oldDocIndex]

        // Decrement frequencies for terms in old document.
        decrementTermFrequencies(for: oldDocument)

        // Replace old document with new one.
        documents[oldDocIndex] = document

        // Add contributions for the new/updated document.
        let tokenizedText = tokenize(document.text)
        documentLengths[document.id] = tokenizedText.count

        incrementTermFrequencies(for: document)

        // Update average document length once.
        updateAverageDocumentLength()
    }

    /// Updates the average document length after changes
    private mutating func updateAverageDocumentLength() {
        guard !documents.isEmpty else {
            self.averageDocumentLength = 0
            return
        }
        let totalLength = documentLengths.values.reduce(0, +)
        self.averageDocumentLength = Float(totalLength) / Float(documents.count)
    }

    /// Increments term frequencies for a document
    /// - Parameter document: The document whose terms should be incremented
    private mutating func incrementTermFrequencies(for document: VecturaDocument) {
        let terms = Set(tokenize(document.text))
        for term in terms {
            documentFrequencies[term, default: 0] += 1
        }
    }

    /// Decrements term frequencies for a document
    /// - Parameter document: The document whose terms should be decremented
    private mutating func decrementTermFrequencies(for document: VecturaDocument) {
        let terms = Set(tokenize(document.text))
        for term in terms {
            if let currentCount = documentFrequencies[term] {
                if currentCount > 1 {
                    documentFrequencies[term] = currentCount - 1
                } else {
                    documentFrequencies.removeValue(forKey: term)
                }
            }
        }
    }

    private func termFrequency(term: String, in document: VecturaDocument) -> Float {
        Float(
            tokenize(document.text)
                .filter { $0 == term }
                .count)
    }
}

extension VecturaDocument {
    /// Calculates a hybrid search score combining vector similarity and BM25
    ///
    /// - Parameters:
    ///   - vectorScore: The vector similarity score
    ///   - bm25Score: The BM25 score
    ///   - weight: Weight for vector score (0.0-1.0), BM25 weight will be (1-weight)
    /// - Returns: Combined score
    public func hybridScore(vectorScore: Float, bm25Score: Float, weight: Float = 0.5) -> Float {
        VecturaDocument.calculateHybridScore(vectorScore: vectorScore, bm25Score: bm25Score, weight: weight)
    }

    /// Calculates a hybrid search score combining vector similarity and BM25
    ///
    /// - Parameters:
    ///   - vectorScore: The vector similarity score
    ///   - bm25Score: The BM25 score
    ///   - weight: Weight for vector score (0.0-1.0), BM25 weight will be (1-weight)
    /// - Returns: Combined score
    public static func calculateHybridScore(vectorScore: Float, bm25Score: Float, weight: Float = 0.5) -> Float {
        let normalizedBM25 = min(max(bm25Score / 10.0, 0), 1)
        return weight * vectorScore + (1 - weight) * normalizedBM25
    }
}
