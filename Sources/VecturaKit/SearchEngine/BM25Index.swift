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
///
/// This is a reference type (class) to avoid expensive copies when modifying the index.
/// Thread safety is provided by the containing BM25SearchEngine actor.
public final class BM25Index: @unchecked Sendable {

  private let k1: Float
  private let b: Float
  private var documents: [UUID: VecturaDocument]
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
    // Use reduce to handle duplicate IDs gracefully (keep last occurrence)
    self.documents = documents.reduce(into: [:]) { dict, doc in
      dict[doc.id] = doc
    }
    self.documentFrequencies = [:]

    // Build documentLengths from deduplicated documents to avoid redundant tokenization
    self.documentLengths = self.documents.reduce(into: [:]) { dict, pair in
      dict[pair.key] = tokenize(pair.value.text).count
    }

    // Guard against division by zero when documents array is empty
    if self.documents.isEmpty {
      self.averageDocumentLength = 0
    } else {
      self.averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(self.documents.count)
    }

    for document in self.documents.values {
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

    for document in documents.values {
      let docLength = Float(documentLengths[document.id] ?? 0)
      var score: Float = 0.0

      // Tokenize document once and reuse for all query terms
      let docTokens = tokenize(document.text)
      let docTokenCounts = Dictionary(grouping: docTokens, by: { $0 }).mapValues { Float($0.count) }

      for term in queryTerms {
        let tf = docTokenCounts[term] ?? 0
        let df = Float(documentFrequencies[term] ?? 0)

        // Ensure argument to log is positive for numerical stability
        let idfArgument = (Float(documents.count) - df + 0.5) / (df + 0.5)
        let idf = log(max(idfArgument, 1e-9))

        let numerator = tf * (k1 + 1)
        let avgDocLen = max(averageDocumentLength, 1e-9)  // Prevent division by zero
        let denominator = tf + k1 * (1 - b + b * docLength / avgDocLen)

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
  public func addDocument(_ document: VecturaDocument) {
    upsertDocument(document)
  }

  /// Remove a document from the index incrementally
  ///
  /// - Parameter documentID: The ID of the document to remove
  public func removeDocument(_ documentID: UUID) {
    guard let document = documents[documentID] else {
      return
    }

    documents.removeValue(forKey: documentID)

    decrementTermFrequencies(for: document)

    documentLengths.removeValue(forKey: documentID)
    updateAverageDocumentLength()
  }

  /// Update an existing document in the index incrementally
  ///
  /// - Parameter document: The updated document
  public func updateDocument(_ document: VecturaDocument) {
    upsertDocument(document)
  }

  /// Internal helper that handles both adding new documents and updating existing ones
  /// - Parameter document: The document to add or update
  private func upsertDocument(_ document: VecturaDocument) {
    // If document already exists, decrement its old term frequencies first
    if let oldDocument = documents[document.id] {
      decrementTermFrequencies(for: oldDocument)
    }

    documents[document.id] = document

    // Tokenize once and reuse for both length and term frequencies
    let tokens = tokenize(document.text)
    let length = tokens.count
    documentLengths[document.id] = length

    let terms = Set(tokens)
    incrementTermFrequencies(terms: terms)

    updateAverageDocumentLength()
  }

  /// Checks if a document with the given ID exists in the index
  /// - Parameter documentID: The document ID to check
  /// - Returns: True if the document exists, false otherwise
  public func containsDocument(withID documentID: UUID) -> Bool {
    documents[documentID] != nil
  }

  /// Updates the average document length after changes
  private func updateAverageDocumentLength() {
    guard !documents.isEmpty else {
      self.averageDocumentLength = 0
      return
    }
    let totalLength = documentLengths.values.reduce(0, +)
    self.averageDocumentLength = Float(totalLength) / Float(documents.count)
  }

  /// Increments term frequencies using pre-computed terms
  /// - Parameter terms: The unique terms to increment
  private func incrementTermFrequencies(terms: Set<String>) {
    for term in terms {
      documentFrequencies[term, default: 0] += 1
    }
  }

  /// Decrements term frequencies for a document
  /// - Parameter document: The document whose terms should be decremented
  private func decrementTermFrequencies(for document: VecturaDocument) {
    let terms = Set(tokenize(document.text))
    decrementTermFrequencies(terms: terms)
  }

  /// Decrements term frequencies using pre-computed terms
  /// - Parameter terms: The unique terms to decrement
  private func decrementTermFrequencies(terms: Set<String>) {
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
}

extension VecturaDocument {

  /// Calculates a hybrid search score combining vector similarity and BM25
  ///
  /// - Parameters:
  ///   - vectorScore: The vector similarity score
  ///   - bm25Score: The BM25 score
  ///   - weight: Weight for vector score (0.0-1.0), BM25 weight will be (1-weight)
  ///   - normalizationFactor: Factor to normalize BM25 scores to 0-1 range
  /// - Returns: Combined score
  public func hybridScore(
    vectorScore: Float,
    bm25Score: Float,
    weight: Float = 0.5,
    normalizationFactor: Float = 10.0
  ) -> Float {
    VecturaDocument.calculateHybridScore(
      vectorScore: vectorScore,
      bm25Score: bm25Score,
      weight: weight,
      normalizationFactor: normalizationFactor
    )
  }

  /// Calculates a hybrid search score combining vector similarity and BM25
  ///
  /// - Parameters:
  ///   - vectorScore: The vector similarity score
  ///   - bm25Score: The BM25 score
  ///   - weight: Weight for vector score (0.0-1.0), BM25 weight will be (1-weight)
  ///   - normalizationFactor: Factor to normalize BM25 scores to 0-1 range
  /// - Returns: Combined score
  public static func calculateHybridScore(
    vectorScore: Float,
    bm25Score: Float,
    weight: Float = 0.5,
    normalizationFactor: Float = 10.0
  ) -> Float {
    let normalizedBM25 = min(max(bm25Score / normalizationFactor, 0), 1)
    return weight * vectorScore + (1 - weight) * normalizedBM25
  }
}
