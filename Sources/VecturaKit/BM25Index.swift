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
        
        if documents.isEmpty {
            self.averageDocumentLength = 0
        } else {
            self.averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(documents.count)
        }
        
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

        // Prevent division by zero if averageDocumentLength is 0
        guard averageDocumentLength > 0 else { return [] }
        
        for document in documents {
            let docLength = Float(documentLengths[document.id] ?? 0)
            var score: Float = 0.0
            
            for term in queryTerms {
                let tf = termFrequency(term: term, in: document)
                let df = Float(documentFrequencies[term] ?? 0)
                
                // Ensure documents.count is not zero for IDF calculation, though averageDocumentLength check above might cover this.
                // Also, ensure df is not equal to documents.count to avoid log(negative) or log(0).
                var idf: Float = 0
                let N = Float(documents.count)
                if N > df && df >= 0 { // df should always be >= 0
                    idf = log((N - df + 0.5) / (df + 0.5))
                } else if N == df && N > 0 { // if term is in all documents
                     idf = log(1.0 / ( (N - df + 0.5) / (df + 0.5) ) ) // A small positive value, effectively
                }


                let numerator = tf * (k1 + 1)
                let denominator = tf + k1 * (1 - b + b * docLength / averageDocumentLength)
                
                if denominator != 0 { // Prevent division by zero
                    score += idf * (numerator / denominator)
                }
            }
            
            scores.append((document, score))
        }
        
        return scores
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .filter { $0.1 > 0 }
    }
    
    /// Add a new document to the index
    ///
    /// - Parameter document: The document to add
    public mutating func addDocument(_ document: VecturaDocument) {
        // Ensure document is not already indexed
        guard !documents.contains(where: { $0.id == document.id }) else {
            // Optionally, update the document if it exists, or simply return
            // For now, returning to prevent duplicate processing
            return
        }

        documents.append(document)
        
        let tokenizedText = tokenize(document.text)
        documentLengths[document.id] = tokenizedText.count
        
        let terms = Set(tokenizedText)
        for term in terms {
            documentFrequencies[term, default: 0] += 1
        }
        
        let totalLength = documentLengths.values.reduce(0, +)
        if documents.isEmpty {
            self.averageDocumentLength = 0
        } else {
            self.averageDocumentLength = Float(totalLength) / Float(documents.count)
        }
    }

    /// Removes a document from the index by its ID.
    ///
    /// - Parameter id: The UUID of the document to remove.
    public mutating func removeDocument(byId id: UUID) {
        guard let index = documents.firstIndex(where: { $0.id == id }) else {
            return // Document not found
        }
        
        let documentToRemove = documents.remove(at: index)
        documentLengths.removeValue(forKey: id)
        
        let termsInRemovedDocument = Set(tokenize(documentToRemove.text))
        for term in termsInRemovedDocument {
            if let currentFreq = documentFrequencies[term] {
                if currentFreq > 1 {
                    documentFrequencies[term] = currentFreq - 1
                } else {
                    documentFrequencies.removeValue(forKey: term)
                }
            }
        }
        
        if documents.isEmpty {
            averageDocumentLength = 0
        } else {
            let totalLength = documentLengths.values.reduce(0, +)
            averageDocumentLength = Float(totalLength) / Float(documents.count)
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
        let normalizedBM25 = min(max(bm25Score / 10.0, 0), 1)
        return weight * vectorScore + (1 - weight) * normalizedBM25
    }
}
