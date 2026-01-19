import Foundation
import VecturaKit

/// Lightweight, deterministic embedder for performance tests.
///
/// This avoids loading Core ML models while still producing stable,
/// repeatable embeddings for accuracy and performance comparisons.
struct DeterministicEmbedder: VecturaEmbedder, Sendable {
  private let dimensionValue: Int

  init(dimension: Int = 384) {
    self.dimensionValue = dimension
  }

  var dimension: Int {
    get async throws { dimensionValue }
  }

  func embed(texts: [String]) async throws -> [[Float]] {
    texts.map { embedding(for: $0) }
  }

  private func embedding(for text: String) -> [Float] {
    var generator = SplitMix64(seed: stableHash(text))
    var values: [Float] = []
    values.reserveCapacity(dimensionValue)

    for _ in 0..<dimensionValue {
      let next = generator.next()
      let normalized = Double(next) / Double(UInt64.max)
      values.append(Float(normalized * 2.0 - 1.0))
    }

    return values
  }

  private func stableHash(_ text: String) -> UInt64 {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in text.utf8 {
      hash ^= UInt64(byte)
      hash &*= 0x100000001b3
    }
    return hash
  }
}

private struct SplitMix64 {
  private var state: UInt64

  init(seed: UInt64) {
    self.state = seed
  }

  mutating func next() -> UInt64 {
    state &+= 0x9e3779b97f4a7c15
    var z = state
    z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
    return z ^ (z >> 31)
  }
}
