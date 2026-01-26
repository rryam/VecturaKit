import Foundation
@testable import VecturaKit

enum PerformanceTestConfig {
  static let useSwiftEmbedder: Bool =
    ProcessInfo.processInfo.environment["VECTURA_PERF_USE_SWIFT_EMBEDDER"] == "1"

  static let defaultDimension = 384

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  static func makeEmbedder(modelSource: VecturaModelSource = .default) -> any VecturaEmbedder {
    if useSwiftEmbedder {
      return SwiftEmbedder(modelSource: modelSource)
    }
    return DeterministicEmbedder(dimensionValue: defaultDimension)
  }

  private struct DeterministicEmbedder: VecturaEmbedder {
    let dimensionValue: Int

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
      var state = stableHash(text)
      var values: [Float] = []
      values.reserveCapacity(dimensionValue)

      for _ in 0..<dimensionValue {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let normalized = Float((state >> 32) & 0xFFFF) / 65535.0
        values.append(normalized * 2 - 1)
      }

      return values
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
}
