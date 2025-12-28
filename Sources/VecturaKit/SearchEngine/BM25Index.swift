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

/// A lightweight document structure for BM25 indexing that doesn't store embeddings.
///
/// This reduces memory usage by storing only what's needed for text search:
/// - Document ID
/// - Text content (for tokenization and results)
/// - Creation timestamp
///
/// Using this instead of full VecturaDocument can reduce memory footprint by
/// ~1.5KB per document (384-dimension embedding) when using indexed memory strategy.
public struct BM25Document: Sendable, Equatable {
  public let id: UUID
  public let text: String
  public let createdAt: Date

  public init(id: UUID, text: String, createdAt: Date) {
    self.id = id
    self.text = text
    self.createdAt = createdAt
  }

  /// Creates a BM25Document from a full VecturaDocument
  public init(from document: VecturaDocument) {
    self.id = document.id
    self.text = document.text
    self.createdAt = document.createdAt
  }
}

/// An index for BM25-based text search using lightweight BM25Document.
///
/// This actor provides thread-safe access to the BM25 index, ensuring proper
/// isolation of mutable state across concurrent operations.
///
/// ## Memory Efficiency
///
/// This implementation uses BM25Document instead of full VecturaDocument to reduce
/// memory usage. Each document only stores:
/// - ID (16 bytes)
/// - Text (variable, typically few hundred bytes)
/// - Timestamp (8 bytes)
///
/// This is significantly smaller than storing full VecturaDocument with embeddings
/// (~1.5KB per document for 384-dimensional embeddings).
public actor BM25Index {

  private let k1: Float
  private let b: Float
  private var documents: [UUID: BM25Document]
  private var documentFrequencies: [String: Int]
  private var documentLengths: [UUID: Int]
  private var documentTokens: [UUID: [String]]
  private var averageDocumentLength: Float

  /// Creates a new BM25 index for the given documents
  ///
  /// - Parameters:
  ///   - documents: The documents to index (converted to BM25Document internally)
  ///   - k1: BM25 k1 parameter (default: 1.2)
  ///   - b: BM25 b parameter (default: 0.75)
  public init(documents: [VecturaDocument], k1: Float = 1.2, b: Float = 0.75) {
    self.k1 = k1
    self.b = b
    // Convert to lightweight BM25Document to reduce memory usage
    let lightweightDocs = documents.map { BM25Document(from: $0) }
    (self.documents, self.documentFrequencies, self.documentLengths, self.documentTokens, self.averageDocumentLength) =
      Self.buildIndex(from: lightweightDocs)
  }

  /// Creates a new BM25 index with lightweight documents
  ///
  /// - Parameters:
  ///   - documents: The lightweight documents to index
  ///   - k1: BM25 k1 parameter (default: 1.2)
  ///   - b: BM25 b parameter (default: 0.75)
  public init(documents: [BM25Document], k1: Float = 1.2, b: Float = 0.75) {
    self.k1 = k1
    self.b = b
    // Use reduce to handle duplicate IDs gracefully (keep last occurrence)
    self.documents = documents.reduce(into: [:]) { dict, doc in
      dict[doc.id] = doc
    }
    self.documentFrequencies = [:]

    // Initialize empty dictionaries first
    var tempTokens: [UUID: [String]] = [:]
    var tempLengths: [UUID: Int] = [:]

    // Tokenize once per document and cache for later reuse
    for (id, document) in self.documents {
      let tokens = tokenize(document.text)
      tempTokens[id] = tokens
      tempLengths[id] = tokens.count
    }

    self.documentTokens = tempTokens
    self.documentLengths = tempLengths

    // Guard against division by zero when documents array is empty
    if self.documents.isEmpty {
      self.averageDocumentLength = 0
    } else {
      self.averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(self.documents.count)
    }

    // Build document frequencies using cached tokens
    for document in self.documents.values {
      let terms = Set(self.documentTokens[document.id] ?? [])
      for term in terms {
        documentFrequencies[term, default: 0] += 1
      }
    }
  }

  // swiftlint:disable large_tuple
  /// Builds the index data structures from documents (non-isolated helper)
  private static func buildIndex(from documents: [BM25Document]) -> (
    [UUID: BM25Document],
    [String: Int],
    [UUID: Int],
    [UUID: [String]],
    Float
  ) {
    // Use reduce to handle duplicate IDs gracefully (keep last occurrence)
    let docsMap = documents.reduce(into: [:]) { dict, doc in
      dict[doc.id] = doc
    }
    var documentFrequencies: [String: Int] = [:]
    var documentLengths: [UUID: Int] = [:]
    var documentTokens: [UUID: [String]] = [:]

    // Tokenize once per document and cache for later reuse
    for (id, document) in docsMap {
      let tokens = tokenize(document.text)
      documentTokens[id] = tokens
      documentLengths[id] = tokens.count
    }

    // Calculate average document length
    let averageDocumentLength: Float
    if docsMap.isEmpty {
      averageDocumentLength = 0
    } else {
      averageDocumentLength = Float(documentLengths.values.reduce(0, +)) / Float(docsMap.count)
    }

    // Build document frequencies using cached tokens
    for document in docsMap.values {
      let terms = Set(documentTokens[document.id] ?? [])
      for term in terms {
        documentFrequencies[term, default: 0] += 1
      }
    }

    return (docsMap, documentFrequencies, documentLengths, documentTokens, averageDocumentLength)
  }
  // swiftlint:enable large_tuple

  /// Searches the index using BM25 scoring
  ///
  /// - Parameters:
  ///   - query: The search query
  ///   - topK: Maximum number of results to return
  /// - Returns: Array of tuples containing lightweight documents and their BM25 scores
  public func search(query: String, topK: Int = 10) -> [(document: BM25Document, score: Float)] {
    let queryTerms = tokenize(query)
    var scores: [(BM25Document, Float)] = []

    for document in documents.values {
      let docLength = Float(documentLengths[document.id] ?? 0)
      var score: Float = 0.0

      // Use cached tokens instead of re-tokenizing
      let docTokens = documentTokens[document.id] ?? []
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
    addDocument(BM25Document(from: document))
  }

  /// Add a lightweight document to the index incrementally
  ///
  /// - Parameter document: The lightweight document to add
  public func addDocument(_ document: BM25Document) {
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
    documentTokens.removeValue(forKey: documentID)
    updateAverageDocumentLength()
  }

  /// Update an existing document in the index incrementally
  ///
  /// - Parameter document: The updated document
  public func updateDocument(_ document: VecturaDocument) {
    updateDocument(BM25Document(from: document))
  }

  /// Update an existing lightweight document in the index incrementally
  ///
  /// - Parameter document: The updated lightweight document
  public func updateDocument(_ document: BM25Document) {
    upsertDocument(document)
  }

  /// Internal helper that handles both adding new documents and updating existing ones
  /// - Parameter document: The document to add or update
  private func upsertDocument(_ document: BM25Document) {
    // If document already exists, decrement its old term frequencies first
    if let oldDocument = documents[document.id] {
      decrementTermFrequencies(for: oldDocument)
    }

    documents[document.id] = document

    // Tokenize once and cache for later reuse
    let tokens = tokenize(document.text)
    documentTokens[document.id] = tokens
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

  /// Clears the entire index, releasing all memory used by document storage.
  ///
  /// This is useful when using indexed memory strategy and wanting to free memory
  /// after search operations. After calling this, the index will need to be rebuilt
  /// before the next search.
  public func unload() {
    documents.removeAll()
    documentFrequencies.removeAll()
    documentLengths.removeAll()
    documentTokens.removeAll()
    averageDocumentLength = 0
  }

  /// Returns the current number of documents in the index
  /// - Returns: The count of indexed documents
  public var documentCount: Int {
    documents.count
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
  private func decrementTermFrequencies(for document: BM25Document) {
    // Use cached tokens if available, otherwise tokenize
    let terms = Set(documentTokens[document.id] ?? tokenize(document.text))
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
