import Foundation
@testable import VecturaKit

struct DeterministicEmbedder: VecturaEmbedder {
  let dimensionValue: Int

  init(dimension: Int = 384) {
    self.dimensionValue = dimension
  }

  var dimension: Int {
    get async throws { dimensionValue }
  }

  func embed(texts: [String]) async throws -> [[Float]] {
    var embeddings: [[Float]] = []
    embeddings.reserveCapacity(texts.count)

    for text in texts {
      embeddings.append(try await embed(text: text))
    }

    return embeddings
  }

  func embed(text: String) async throws -> [Float] {
    var vector = [Float](repeating: 0, count: dimensionValue)
    let tokens = text
      .lowercased()
      .split { !$0.isLetter && !$0.isNumber }

    for token in tokens {
      let index = Int(stableHash(String(token)) % UInt64(dimensionValue))
      vector[index] += 1
    }

    return try VectorMath.normalizeEmbedding(vector)
  }

  private func stableHash(_ text: String) -> UInt64 {
    var hash: UInt64 = 1469598103934665603
    for byte in text.utf8 {
      hash ^= UInt64(byte)
      hash &*= 1099511628211
    }
    return hash == 0 ? 1 : hash
  }
}
