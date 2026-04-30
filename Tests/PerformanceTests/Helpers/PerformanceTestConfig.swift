import Foundation
@testable import VecturaKit
@testable import VecturaNLKit

enum PerformanceTestConfig {
  private static let environment = ProcessInfo.processInfo.environment

  static let useNLEmbedder: Bool =
    environment["VECTURA_PERF_USE_NL_EMBEDDER"] == "1"

  static let performanceProfile: String =
    (environment["VECTURA_PERF_PROFILE"] ?? "default").lowercased()

  static let runRealisticBenchmarks: Bool =
    performanceProfile == "realistic" || environment["VECTURA_PERF_REALISTIC"] == "1"

  static let defaultDimension = 384
  static let realisticDocumentCount =
    intEnv("VECTURA_PERF_REALISTIC_DOCS", default: useNLEmbedder ? 2_500 : 12_000)
  static let realisticQueryCount =
    intEnv("VECTURA_PERF_REALISTIC_QUERIES", default: useNLEmbedder ? 160 : 600)
  static let realisticMixedOperationCount =
    intEnv("VECTURA_PERF_REALISTIC_MIXED_OPS", default: useNLEmbedder ? 240 : 1_200)
  static let realisticConcurrentClients =
    intEnv("VECTURA_PERF_REALISTIC_CLIENTS", default: useNLEmbedder ? 4 : 12)
  static let realisticColdRuns =
    intEnv("VECTURA_PERF_REALISTIC_COLD_RUNS", default: useNLEmbedder ? 6 : 24)

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  static func makeEmbedder() async throws -> any VecturaEmbedder {
    if useNLEmbedder {
      return try await NLContextualEmbedder(language: .english)
    }
    return DeterministicEmbedder(dimensionValue: defaultDimension)
  }

  private static func intEnv(_ key: String, default defaultValue: Int) -> Int {
    guard let raw = environment[key], let value = Int(raw), value > 0 else {
      return defaultValue
    }
    return value
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
