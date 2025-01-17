import Foundation
import MLX
import MLXEmbedders

/// A vector database implementation that stores and searches documents using their vector embeddings.
public class VecturaKit: VecturaProtocol {

  /// The configuration for this vector database instance.
  private let config: VecturaConfig

  /// The storage for documents.
  private var documents: [UUID: VecturaDocument]

  /// The storage directory for documents.
  private let storageDirectory: URL

  /// Cached normalized embeddings for faster search
  private var normalizedEmbeddings: [UUID: MLXArray] = [:]

  /// Creates a new vector database instance.
  ///
  /// - Parameter config: The configuration for the database.
  public init(config: VecturaConfig) throws {
    self.config = config
    self.documents = [:]

    /// Create storage directory in the app's Documents directory
    self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
      .first!
      .appendingPathComponent("VecturaKit")
      .appendingPathComponent(config.name)

    /// Try to create directory and load existing documents
    try FileManager.default.createDirectory(
      at: storageDirectory,
      withIntermediateDirectories: true
    )

    try loadDocuments()
  }

  /// Adds multiple documents to the vector store in batch.
  ///
  /// - Parameters:
  ///   - texts: Array of text contents to add
  ///   - ids: Optional array of unique identifiers (must match texts.count if provided)
  ///   - modelConfig: The model configuration to use (default: nomic_text_v1_5)
  /// - Returns: Array of IDs for the added documents
  public func addDocuments(
    texts: [String],
    ids: [UUID]? = nil,
    modelConfig: ModelConfiguration = .nomic_text_v1_5
  ) async throws -> [UUID] {
    // Validate ids if provided
    if let ids = ids {
      guard ids.count == texts.count else {
        throw VecturaError.invalidInput("Number of IDs must match number of texts")
      }
    }

    // Create embeddings for all texts in one batch
    let embeddings = try await createEmbeddings(for: texts, modelConfig: modelConfig)

    // Validate dimensions for all embeddings
    for embedding in embeddings {
      guard embedding.shape.last == config.dimension else {
        throw VecturaError.dimensionMismatch(
          expected: config.dimension,
          got: embedding.shape.last ?? 0
        )
      }
    }

    // Create documents and save them
    var documentIds: [UUID] = []
    let documentsToSave = zip(texts, embeddings).enumerated().map { index, pair in
      let (text, embedding) = pair
      let id = ids?[index] ?? UUID()
      documentIds.append(id)

      return VecturaDocument(
        id: id,
        text: text,
        embedding: embedding
      )
    }

    // Pre-compute normalized embeddings for search
    for document in documentsToSave {
      let norm = sqrt(sum(document.embedding * document.embedding))
      normalizedEmbeddings[document.id] = document.embedding / norm
      documents[document.id] = document
    }

    // Save all documents in parallel
    try await withThrowingTaskGroup(of: Void.self) { group in
      for document in documentsToSave {
        group.addTask {
          try await self.saveDocument(document)
        }
      }
      try await group.waitForAll()
    }

    return documentIds
  }

  /// Adds a document to the vector store.
  ///
  /// - Parameters:
  ///   - text: The text content of the document.
  ///   - id: Optional unique identifier for the document.
  ///   - modelConfig: The model configuration to use (default: nomic_text_v1_5)
  /// - Returns: The ID of the added document.
  public func addDocument(
    text: String,
    id: UUID? = nil,
    modelConfig: ModelConfiguration = .nomic_text_v1_5
  ) async throws -> UUID {
    let ids = try await addDocuments(
      texts: [text],
      ids: id.map { [$0] },
      modelConfig: modelConfig
    )
    return ids[0]
  }

  /// Searches for similar documents using a query vector.
  ///
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

    let queryNorm = sqrt(sum(query * query))
    let normalizedQuery = query / queryNorm

    var results: [VecturaSearchResult] = []

    for document in documents.values {
      let normalizedVector = normalizedEmbeddings[document.id]!
      let similarity = sum(normalizedQuery * normalizedVector)
        .asArray(Float.self)[0]

      if let minThreshold = threshold ?? config.searchOptions.minThreshold,
        similarity < minThreshold
      {
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

    results.sort { $0.score > $1.score }
    let limit = numResults ?? config.searchOptions.defaultNumResults
    return Array(results.prefix(limit))
  }

  /// Resets the vector database.
  public func reset() async throws {
    documents.removeAll()
    normalizedEmbeddings.removeAll()

    /// Remove all files from storage
    let fileURLs = try FileManager.default.contentsOfDirectory(
      at: storageDirectory,
      includingPropertiesForKeys: nil
    )

    for fileURL in fileURLs {
      try FileManager.default.removeItem(at: fileURL)
    }
  }

  /// Saves a document to storage.
  private func saveDocument(_ document: VecturaDocument) async throws {
    let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    let data = try encoder.encode(document)
    try data.write(to: documentURL)
  }

  /// Loads existing documents from storage.
  private func loadDocuments() throws {
    let fileURLs = try FileManager.default.contentsOfDirectory(
      at: storageDirectory,
      includingPropertiesForKeys: nil
    )

    let decoder = JSONDecoder()
    var loadErrors: [String] = []

    for fileURL in fileURLs where fileURL.pathExtension == "json" {
      do {
        let data = try Data(contentsOf: fileURL)
        let document = try decoder.decode(VecturaDocument.self, from: data)
        documents[document.id] = document
      } catch {
        loadErrors.append("Failed to load document at \(fileURL): \(error.localizedDescription)")
      }
    }

    if !loadErrors.isEmpty {
      throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
    }
  }

  /// Creates embeddings for the given texts using the specified model configuration.
  /// - Parameters:
  ///   - texts: Array of texts to create embeddings for
  ///   - modelConfig: The model configuration to use (default: nomic_text_v1_5)
  /// - Returns: Array of embeddings as MLXArray
  public func createEmbeddings(
    for texts: [String],
    modelConfig: ModelConfiguration = .nomic_text_v1_5
  ) async throws -> [MLXArray] {
    let modelContainer = try await MLXEmbedders.loadModelContainer(
      configuration: modelConfig)

    let embeddings = await modelContainer.perform {
      (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in

      let inputs = texts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      let maxLength = inputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }

      let padded = stacked(
        inputs.map { elem in
          MLXArray(
            elem
              + Array(
                repeating: tokenizer.eosTokenId ?? 0,
                count: maxLength - elem.count))
        })

      let mask = (padded .!= tokenizer.eosTokenId ?? 0)
      let tokenTypes = MLXArray.zeros(like: padded)

      let result = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
      )

      return result.map { $0.asArray(Float.self) }
    }

    return embeddings.map { MLXArray($0) }
  }
}
