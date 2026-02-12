import CoreML
import Embeddings
import Foundation

/// An embedder implementation using swift-embeddings library
/// (Bert, NomicBert, Model2Vec, and StaticEmbeddings models).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public actor SwiftEmbedder {

  /// Configuration options for `SwiftEmbedder`.
  public struct Configuration: Sendable {
    /// Optional output dimension for StaticEmbeddings models.
    ///
    /// When set, embeddings are truncated to this dimension (capped at the model dimension).
    /// Values less than 1 are rejected with `VecturaError.invalidInput`.
    public var staticEmbeddingsTruncateDimension: Int?

    public init(staticEmbeddingsTruncateDimension: Int? = nil) {
      self.staticEmbeddingsTruncateDimension = staticEmbeddingsTruncateDimension
    }
  }

  enum ResolvedModelFamily: Sendable, Equatable {
    case bert
    case model2vec
    case staticEmbeddings
    case nomicBert
  }

  private let modelSource: VecturaModelSource
  private let configuration: Configuration
  private var bertModel: Bert.ModelBundle?
  private var nomicBertModel: NomicBert.ModelBundle?
  private var model2vecModel: Model2Vec.ModelBundle?
  private var staticEmbeddingsModel: StaticEmbeddings.ModelBundle?
  private var cachedDimension: Int?

  /// Initializes a SwiftEmbedder with the specified model source.
  ///
  /// - Parameter modelSource: The source from which to load the embedding model.
  /// - Parameter configuration: Additional embedder behavior configuration.
  public init(
    modelSource: VecturaModelSource = .default,
    configuration: Configuration = .init()
  ) {
    self.modelSource = modelSource
    self.configuration = configuration
  }

  static func resolveModelFamily(for source: VecturaModelSource) -> ResolvedModelFamily {
    switch source {
    case .id(_, let type), .folder(_, let type):
      if let type {
        switch type {
        case .bert: return .bert
        case .model2vec: return .model2vec
        case .staticEmbeddings: return .staticEmbeddings
        case .nomicBert: return .nomicBert
        }
      }
    }

    let modelId = source.description.lowercased()
    if modelId.contains("minishlab") ||
        modelId.contains("potion") ||
        modelId.contains("model2vec") ||
        modelId.contains("m2v")
    {
      return .model2vec
    }

    if modelId.contains("static-retrieval") ||
        modelId.contains("static-similarity") ||
        modelId.contains("static-embed")
    {
      return .staticEmbeddings
    }

    if modelId.contains("nomic-embed-text") {
      return .nomicBert
    }

    return .bert
  }

  static func resolvedStaticEmbeddingDimension(
    baseDimension: Int,
    truncateDimension: Int?
  ) throws -> Int {
    guard let truncateDimension else {
      return baseDimension
    }

    guard truncateDimension > 0 else {
      throw VecturaError.invalidInput(
        "StaticEmbeddings truncateDimension must be greater than 0, got \(truncateDimension)"
      )
    }

    return min(baseDimension, truncateDimension)
  }

  private func staticEmbeddingsTruncateDimension() throws -> Int? {
    guard let truncateDimension = configuration.staticEmbeddingsTruncateDimension else {
      return nil
    }

    guard truncateDimension > 0 else {
      throw VecturaError.invalidInput(
        "StaticEmbeddings truncateDimension must be greater than 0, got \(truncateDimension)"
      )
    }

    return truncateDimension
  }
}

// MARK: - VecturaEmbedder

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension SwiftEmbedder: VecturaEmbedder {

  /// The dimensionality of the embedding vectors produced by this embedder.
  ///
  /// This value is cached after first detection to avoid repeated computation.
  /// - Throws: An error if the dimension cannot be determined.
  public var dimension: Int {
    get async throws {
      if let cached = cachedDimension {
        return cached
      }

      // Ensure model is loaded
      try await ensureModelLoaded()
      let staticTruncateDimension = try staticEmbeddingsTruncateDimension()

      let dim: Int
      if let model2vec = model2vecModel {
        // Note: 'dimienstion' is a typo in the upstream swift-embeddings library
        // See: swift-embeddings/Sources/Embeddings/Model2Vec/Model2VecModel.swift
        dim = model2vec.model.dimienstion
      } else if let staticEmbeddings = staticEmbeddingsModel {
        dim = try Self.resolvedStaticEmbeddingDimension(
          baseDimension: staticEmbeddings.model.dimension,
          truncateDimension: staticTruncateDimension
        )
      } else if let nomicBert = nomicBertModel {
        let testEmbedding = try nomicBert.encode("test")
        guard let lastDim = testEmbedding.shape.last else {
          throw VecturaError.invalidInput(
            "Could not determine NomicBert model dimension from shape \(testEmbedding.shape)"
          )
        }
        dim = lastDim
      } else if let bert = bertModel {
        // For BERT, we need to get dimension from a test encoding
        let testEmbedding = try bert.encode("test")
        guard let lastDim = testEmbedding.shape.last else {
          throw VecturaError.invalidInput(
            "Could not determine BERT model dimension from shape \(testEmbedding.shape)"
          )
        }
        dim = lastDim
      } else {
        throw VecturaError.invalidInput("No model loaded to detect dimension")
      }

      cachedDimension = dim
      return dim
    }
  }

  /// Generates embeddings for multiple texts in batch.
  ///
  /// - Parameter texts: The text strings to embed.
  /// - Returns: An array of embedding vectors, one for each input text.
  /// - Throws: An error if embedding generation fails.
  public func embed(texts: [String]) async throws -> [[Float]] {
    try await ensureModelLoaded()
    let staticTruncateDimension = try staticEmbeddingsTruncateDimension()

    let embeddingsTensor: MLTensor
    if let model2vec = model2vecModel {
      embeddingsTensor = try model2vec.batchEncode(texts)
    } else if let staticEmbeddings = staticEmbeddingsModel {
      embeddingsTensor = try staticEmbeddings.batchEncode(
        texts,
        normalize: true,
        truncateDimension: staticTruncateDimension
      )
    } else if let nomicBert = nomicBertModel {
      embeddingsTensor = try nomicBert.batchEncode(texts)
    } else if let bert = bertModel {
      embeddingsTensor = try bert.batchEncode(texts)
    } else {
      throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
    }

    let shape = embeddingsTensor.shape
    guard shape.count == 2, let dimension = shape.last else {
      throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
    }
    let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(of: Float.self)
    let allScalars = embeddingShapedArray.scalars

    return stride(from: 0, to: allScalars.count, by: dimension).map {
      Array(allScalars[$0..<($0 + dimension)])
    }
  }

  /// Generates an embedding for a single text.
  ///
  /// - Parameter text: The text string to embed.
  /// - Returns: The embedding vector for the input text.
  /// - Throws: An error if embedding generation fails.
  public func embed(text: String) async throws -> [Float] {
    try await ensureModelLoaded()
    let staticTruncateDimension = try staticEmbeddingsTruncateDimension()

    let embeddingTensor: MLTensor
    if let model2vec = model2vecModel {
      embeddingTensor = try model2vec.encode(text)
    } else if let staticEmbeddings = staticEmbeddingsModel {
      embeddingTensor = try staticEmbeddings.encode(
        text,
        normalize: true,
        truncateDimension: staticTruncateDimension
      )
    } else if let nomicBert = nomicBertModel {
      embeddingTensor = try nomicBert.encode(text)
    } else if let bert = bertModel {
      embeddingTensor = try bert.encode(text)
    } else {
      throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
    }

    let embeddingShapedArray = await embeddingTensor.cast(to: Float.self).shapedArray(of: Float.self)
    return embeddingShapedArray.scalars
  }

  private func ensureModelLoaded() async throws {
    guard bertModel == nil &&
        nomicBertModel == nil &&
        model2vecModel == nil &&
        staticEmbeddingsModel == nil
    else {
      return
    }

    switch Self.resolveModelFamily(for: modelSource) {
    case .model2vec:
      model2vecModel = try await Model2Vec.loadModelBundle(from: modelSource)
    case .staticEmbeddings:
      staticEmbeddingsModel = try await StaticEmbeddings.loadModelBundle(from: modelSource)
    case .nomicBert:
      nomicBertModel = try await NomicBert.loadModelBundle(from: modelSource)
    case .bert:
      bertModel = try await Bert.loadModelBundle(from: modelSource)
    }
  }
}

// MARK: - Model Loading Extensions

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {

  static func loadModelBundle(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
    switch source {
    case .id(let modelId, _):
      do {
        return try await loadModelBundle(from: modelId)
      } catch {
        // Some BERT checkpoints (for example, google-bert/bert-base-uncased)
        // require alternative key mapping.
        return try await loadModelBundle(from: modelId, loadConfig: .googleBert)
      }
    case .folder(let url, _):
      return try await loadModelBundle(from: url)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Model2Vec {

  static func loadModelBundle(from source: VecturaModelSource) async throws -> Model2Vec.ModelBundle {
    switch source {
    case .id(let modelId, _):
      try await loadModelBundle(from: modelId)
    case .folder(let url, _):
      try await loadModelBundle(from: url)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {

  static func loadModelBundle(from source: VecturaModelSource) async throws -> StaticEmbeddings.ModelBundle {
    switch source {
    case .id(let modelId, _):
      try await loadModelBundle(from: modelId, loadConfig: .staticEmbeddings)
    case .folder(let url, _):
      try await loadModelBundle(from: url, loadConfig: .staticEmbeddings)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {

  static func loadModelBundle(from source: VecturaModelSource) async throws -> NomicBert.ModelBundle {
    switch source {
    case .id(let modelId, _):
      try await loadModelBundle(from: modelId)
    case .folder(let url, _):
      try await loadModelBundle(from: url)
    }
  }
}
