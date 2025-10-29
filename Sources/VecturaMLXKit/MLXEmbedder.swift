import Foundation
import MLX
import MLXEmbedders
import VecturaKit

public actor MLXEmbedder: VecturaEmbedder {
  private let modelContainer: ModelContainer
  private let configuration: ModelConfiguration
  private var cachedDimension: Int?

  public init(configuration: ModelConfiguration = .nomic_text_v1_5) async throws {
    self.configuration = configuration
    self.modelContainer = try await MLXEmbedders.loadModelContainer(configuration: configuration)
  }

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

  public func embed(texts: [String]) async throws -> [[Float]] {
    await modelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in
      let inputs = texts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      // Pad to longest
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
  }
}
