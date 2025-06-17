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
    /// Number of documents that contain each term
    private var documentFrequencies: [String: Int]
    /// Length of each document in tokens
    private var documentLengths: [UUID: Int]
    /// Pre-computed term frequencies for each document
    private var termFrequencies: [UUID: [String: Int]]
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
        self.documentLengths = [:]
        self.termFrequencies = [:]
        
        // Pre-compute term frequencies and document lengths in one pass
        for document in documents {
            let tokens = tokenize(document.text)
            let docLength = tokens.count
            documentLengths[document.id] = docLength
            
            // Build term frequency map for this document
            var docTermFreqs: [String: Int] = [:]
            for token in tokens {
                docTermFreqs[token, default: 0] += 1
            }
            termFrequencies[document.id] = docTermFreqs
            
            // Update document frequencies (number of docs containing each term)
            let uniqueTerms = Set(docTermFreqs.keys)
            for term in uniqueTerms {
                documentFrequencies[term, default: 0] += 1
            }
        }
        
        self.averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(documents.count)
    }
    
    /// Searches the index using BM25 scoring
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - topK: Maximum number of results to return
    /// - Returns: Array of tuples containing documents and their BM25 scores
    public func search(query: String, topK: Int = 10) -> [(document: VecturaDocument, score: Float)] {
        let queryTerms = tokenize(query)
        guard !queryTerms.isEmpty else { return [] }
        
        var scores: [(VecturaDocument, Float)] = []
        scores.reserveCapacity(documents.count)
        
        for document in documents {
            let docLength = Float(documentLengths[document.id] ?? 0)
            guard let docTermFreqs = termFrequencies[document.id] else { continue }
            
            var score: Float = 0.0
            
            for term in queryTerms {
                let tf = Float(docTermFreqs[term] ?? 0)
                guard tf > 0 else { continue } // Skip terms not in document
                
                let df = Float(documentFrequencies[term] ?? 0)
                guard df > 0 else { continue } // Skip terms not in any document
                
                let idf = log((Float(documents.count) - df + 0.5) / (df + 0.5))
                let numerator = tf * (k1 + 1)
                let denominator = tf + k1 * (1 - b + b * docLength / averageDocumentLength)
                
                score += idf * (numerator / denominator)
            }
            
            if score > 0 {
                scores.append((document, score))
            }
        }
        
        return scores
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { $0 }
    }
    
    /// Add a new document to the index
    ///
    /// - Parameter document: The document to add
    public mutating func addDocument(_ document: VecturaDocument) {
        documents.append(document)
        
        let tokens = tokenize(document.text)
        let docLength = tokens.count
        documentLengths[document.id] = docLength
        
        // Build term frequency map for this document
        var docTermFreqs: [String: Int] = [:]
        for token in tokens {
            docTermFreqs[token, default: 0] += 1
        }
        termFrequencies[document.id] = docTermFreqs
        
        // Update document frequencies (number of docs containing each term)
        let uniqueTerms = Set(docTermFreqs.keys)
        for term in uniqueTerms {
            documentFrequencies[term, default: 0] += 1
        }
        
        // Recalculate average document length
        let totalLength = documentLengths.values.reduce(0, +)
        self.averageDocumentLength = Float(totalLength) / Float(documents.count)
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
