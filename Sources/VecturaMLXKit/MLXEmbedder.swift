import Foundation
import MLX
import MLXEmbedders
import VecturaKit

/// An embedder implementation using MLX library for generating vector embeddings.
public actor MLXEmbedder: VecturaEmbedder {
  private let modelContainer: ModelContainer
  private let configuration: ModelConfiguration
  private var cachedDimension: Int?

  /// Initializes an MLXEmbedder with the specified model configuration.
  ///
  /// - Parameter configuration: The MLX model configuration to use. Defaults to `.nomic_text_v1_5`.
  /// - Throws: An error if the model container cannot be loaded.
  public init(configuration: ModelConfiguration = .nomic_text_v1_5) async throws {
    self.configuration = configuration
    self.modelContainer = try await MLXEmbedders.loadModelContainer(configuration: configuration)
  }

  /// The dimensionality of the embedding vectors produced by this embedder.
  ///
  /// This value is cached after first detection to avoid repeated computation.
  /// - Throws: An error if the dimension cannot be determined.
  public var dimension: Int {
    get async throws {
      if let cached = cachedDimension {
        return cached
      }

      // Detect dimension by encoding a test string
      let testEmbedding = try await embed(text: "test")
      let dim = testEmbedding.count
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
    guard !texts.isEmpty else {
      throw VecturaError.invalidInput("Cannot embed empty array of texts")
    }

    for (index, text) in texts.enumerated() {
      guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw VecturaError.invalidInput("Text at index \(index) cannot be empty or whitespace-only")
      }
    }

    return try await modelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in
      let inputs = texts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      // Determine padding token
      guard let padToken = tokenizer.eosTokenId else {
        throw EmbeddingError.noPaddingToken
      }

      // Calculate actual max length from inputs
      let maxLength = inputs.map { $0.count }.max() ?? 0

      let padded = stacked(
        inputs.map { tokens in
          MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
        })

      let mask = (padded .!= padToken)
      let tokenTypes = MLXArray.zeros(like: padded)

      // Call model to get outputs
      let outputs = model(
        padded,
        positionIds: nil,
        tokenTypeIds: tokenTypes,
        attentionMask: mask
      )

      // Apply pooling with mask explicitly
      let pooled = pooling(
        outputs,
        mask: mask,
        normalize: true,
        applyLayerNorm: true
      )
      pooled.eval()

      // Handle both 2D [batch, dim] and 3D [batch, seq, dim] shapes
      let finalEmbeddings: MLXArray
      switch pooled.ndim {
      case 2:
        // Expected shape: [batch, dimension]
        finalEmbeddings = pooled

      case 3:
        // Fallback: pooling returned sequence embeddings [batch, seq, dim]
        // Apply mean pooling over sequence dimension
        finalEmbeddings = mean(pooled, axis: 1)
        finalEmbeddings.eval()

      default:
        throw EmbeddingError.unsupportedPoolingShape(pooled.shape)
      }

      let vectors = finalEmbeddings.map { $0.asArray(Float.self) }

      guard vectors.count == texts.count else {
        throw EmbeddingError.vectorCountMismatch(
          expected: texts.count,
          received: vectors.count
        )
      }

      return vectors
    }
  }
}

enum EmbeddingError: Error {
  case noPaddingToken
  case unsupportedPoolingShape([Int])
  case vectorCountMismatch(expected: Int, received: Int)
}
