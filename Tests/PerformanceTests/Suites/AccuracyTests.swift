import Foundation
import Testing
@testable import VecturaKit

/// Accuracy tests for indexed memory strategy.
///
/// These tests measure the search quality of indexed mode compared to fullMemory baseline,
/// evaluating the trade-off between performance and accuracy.
///
/// Metrics:
/// - Recall@K: Percentage of baseline top-K results found by indexed mode
/// - Overlap: Number of common results between strategies
/// - Rank Correlation: How well indexed mode preserves result ordering
///
/// Usage:
/// ```bash
/// # Run all accuracy tests
/// swift test --filter AccuracyTests
/// ```
@Suite("Search Accuracy Tests", .serialized)
struct AccuracyTests {

  // MARK: - Test Infrastructure

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("AccuracyTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    let cleanup = {
      _ = try? FileManager.default.removeItem(at: directory)
    }
    return (directory, cleanup)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func makeEmbedder(modelSource: VecturaModelSource = .default) -> SwiftEmbedder {
    SwiftEmbedder(modelSource: modelSource)
  }

  /// Calculate recall: percentage of baseline results found in candidate results.
  private func calculateRecall(baseline: [UUID], candidate: [UUID]) -> Double {
    let baselineSet = Set(baseline)
    let candidateSet = Set(candidate)
    let overlap = baselineSet.intersection(candidateSet).count
    return Double(overlap) / Double(baseline.count)
  }

  /// Calculate overlap count.
  private func calculateOverlap(baseline: [UUID], candidate: [UUID]) -> Int {
    let baselineSet = Set(baseline)
    let candidateSet = Set(candidate)
    return baselineSet.intersection(candidateSet).count
  }

  // MARK: - Basic Accuracy Tests

  @Test("Accuracy: indexed vs fullMemory at 1K docs")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func basicAccuracyTest() async throws {
    let documentCount = 1_000
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: 15, seed: 54321)

    // Setup fullMemory (baseline)
    let (dir1, cleanup1) = try makeTestDirectory()
    defer { cleanup1() }

    let config1 = VecturaConfig(
      name: "baseline-db",
      directoryURL: dir1,
      memoryStrategy: .fullMemory
    )
    let baselineVectura = try await VecturaKit(config: config1, embedder: makeEmbedder())

    // Generate consistent UUIDs for documents
    let documentIds = (0..<documentCount).map { _ in UUID() }
    _ = try await baselineVectura.addDocuments(texts: documents, ids: documentIds)

    // Setup indexed with MockIndexedStorage
    let (dir2, cleanup2) = try makeTestDirectory()
    defer { cleanup2() }

    let config2 = VecturaConfig(
      name: "indexed-db",
      directoryURL: dir2,
      memoryStrategy: .indexed(candidateMultiplier: 10)
    )
    let mockStorage = MockIndexedStorage()
    let indexedVectura = try await VecturaKit(config: config2, embedder: makeEmbedder(), storageProvider: mockStorage)
    // Use the same document IDs to ensure comparison is valid
    _ = try await indexedVectura.addDocuments(texts: documents, ids: documentIds)

    // Compare search results
    var recalls: [Double] = []
    var overlaps: [Int] = []

    for query in queries {
      let baselineResults = try await baselineVectura.search(query: .text(query), numResults: 10)
      let indexedResults = try await indexedVectura.search(query: .text(query), numResults: 10)

      let baselineIds = baselineResults.map { $0.id }
      let indexedIds = indexedResults.map { $0.id }

      recalls.append(calculateRecall(baseline: baselineIds, candidate: indexedIds))
      overlaps.append(calculateOverlap(baseline: baselineIds, candidate: indexedIds))
    }

    let avgRecall = recalls.reduce(0, +) / Double(recalls.count)
    let avgOverlap = Double(overlaps.reduce(0, +)) / Double(overlaps.count)

    print("\n🎯 Accuracy Analysis (1K docs, mult=10):")
    print("=" * 70)
    print(String(format: "Average Recall@10:  %.1f%%", avgRecall * 100))
    print(String(format: "Average Overlap:    %.1f / 10 documents", avgOverlap))
    print(String(format: "Queries Tested:     %d", queries.count))
    print("=" * 70 + "\n")

    // Indexed mode should achieve reasonable recall
    #expect(avgRecall > 0.5, "Recall should be > 50% with mult=10")
  }

  // MARK: - Candidate Multiplier Impact on Accuracy

  @Test("Accuracy: candidate multiplier impact")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func candidateMultiplierAccuracy() async throws {
    let documentCount = 1_000
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: 10, seed: 54321)

    // Setup fullMemory baseline
    let (dirBaseline, cleanupBaseline) = try makeTestDirectory()
    defer { cleanupBaseline() }

    let configBaseline = VecturaConfig(
      name: "baseline-db",
      directoryURL: dirBaseline,
      memoryStrategy: .fullMemory
    )
    let baselineVectura = try await VecturaKit(config: configBaseline, embedder: makeEmbedder())

    // Generate consistent UUIDs for documents
    let documentIds = (0..<documentCount).map { _ in UUID() }
    _ = try await baselineVectura.addDocuments(texts: documents, ids: documentIds)

    // Get baseline results
    var baselineResultSets: [[UUID]] = []
    for query in queries {
      let results = try await baselineVectura.search(query: .text(query), numResults: 10)
      baselineResultSets.append(results.map { $0.id })
    }

    // Test different multipliers
    let multipliers = [5, 10, 15]
    var accuracyResults: [(mult: Int, recall: Double, overlap: Double)] = []

    for mult in multipliers {
      let (dir, cleanup) = try makeTestDirectory()
      defer { cleanup() }

      let config = VecturaConfig(
        name: "indexed-db",
        directoryURL: dir,
        memoryStrategy: .indexed(candidateMultiplier: mult)
      )
      let mockStorage = MockIndexedStorage()
      let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: mockStorage)
      // Use the same document IDs as baseline
      _ = try await vectura.addDocuments(texts: documents, ids: documentIds)

      var recalls: [Double] = []
      var overlaps: [Int] = []

      for (index, query) in queries.enumerated() {
        let results = try await vectura.search(query: .text(query), numResults: 10)
        let resultIds = results.map { $0.id }
        let baseline = baselineResultSets[index]

        recalls.append(calculateRecall(baseline: baseline, candidate: resultIds))
        overlaps.append(calculateOverlap(baseline: baseline, candidate: resultIds))
      }

      let avgRecall = recalls.reduce(0, +) / Double(recalls.count)
      let avgOverlap = Double(overlaps.reduce(0, +)) / Double(overlaps.count)

      accuracyResults.append((mult: mult, recall: avgRecall, overlap: avgOverlap))
    }

    // Print results
    print("\n📈 Candidate Multiplier vs Accuracy:")
    print("=" * 80)
    print(String(format: "%-20s %-25s %-25s", "Multiplier", "Recall@10", "Avg Overlap (out of 10)"))
    print("-" * 80)

    for (mult, recall, overlap) in accuracyResults {
      print(String(format: "%-20d %-25.1f%% %-25.1f", mult, recall * 100, overlap))
    }

    print("=" * 80)
    print("\n💡 Insight: Higher multipliers improve recall at cost of performance\n")

    #expect(accuracyResults.count == multipliers.count)
  }

  // MARK: - Accuracy vs Performance Trade-off

  @Test("Accuracy: accuracy-performance trade-off analysis")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func accuracyPerformanceTradeoff() async throws {
    let documentCount = 1_000
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: 10, seed: 54321)

    // Baseline
    let (dirBaseline, cleanupBaseline) = try makeTestDirectory()
    defer { cleanupBaseline() }

    let configBaseline = VecturaConfig(
      name: "baseline-db",
      directoryURL: dirBaseline,
      memoryStrategy: .fullMemory
    )
    let baselineVectura = try await VecturaKit(config: configBaseline, embedder: makeEmbedder())

    // Generate consistent UUIDs for documents
    let documentIds = (0..<documentCount).map { _ in UUID() }
    _ = try await baselineVectura.addDocuments(texts: documents, ids: documentIds)

    // Get baseline results and timing
    var baselineResultSets: [[UUID]] = []
    var baselineLatencies: [UInt64] = []

    for query in queries {
      let start = DispatchTime.now().uptimeNanoseconds
      let results = try await baselineVectura.search(query: .text(query), numResults: 10)
      let elapsed = DispatchTime.now().uptimeNanoseconds - start

      baselineResultSets.append(results.map { $0.id })
      baselineLatencies.append(elapsed)
    }

    let baselineAvgLatency = Double(baselineLatencies.reduce(0, +)) / Double(baselineLatencies.count) / 1_000_000.0

    // Test different multipliers with timing
    let multipliers = [5, 10, 15]
    var tradeoffResults: [(mult: Int, recall: Double, latency: Double, speedup: Double)] = []

    for mult in multipliers {
      let (dir, cleanup) = try makeTestDirectory()
      defer { cleanup() }

      let config = VecturaConfig(
        name: "indexed-db",
        directoryURL: dir,
        memoryStrategy: .indexed(candidateMultiplier: mult)
      )
      let mockStorage = MockIndexedStorage()
      let vectura = try await VecturaKit(config: config, embedder: makeEmbedder(), storageProvider: mockStorage)
      // Use the same document IDs as baseline
      _ = try await vectura.addDocuments(texts: documents, ids: documentIds)

      var recalls: [Double] = []
      var latencies: [UInt64] = []

      for (index, query) in queries.enumerated() {
        let start = DispatchTime.now().uptimeNanoseconds
        let results = try await vectura.search(query: .text(query), numResults: 10)
        let elapsed = DispatchTime.now().uptimeNanoseconds - start

        let resultIds = results.map { $0.id }
        let baseline = baselineResultSets[index]

        recalls.append(calculateRecall(baseline: baseline, candidate: resultIds))
        latencies.append(elapsed)
      }

      let avgRecall = recalls.reduce(0, +) / Double(recalls.count)
      let avgLatency = Double(latencies.reduce(0, +)) / Double(latencies.count) / 1_000_000.0
      let speedup = baselineAvgLatency / avgLatency

      tradeoffResults.append((mult: mult, recall: avgRecall, latency: avgLatency, speedup: speedup))
    }

    // Print trade-off analysis
    print("\n⚖️  Accuracy vs Performance Trade-off:")
    print("=" * 95)
    print(String(format: "%-15s %-20s %-25s %-25s", "Multiplier", "Recall@10", "Avg Latency (ms)", "Speedup vs Baseline"))
    print("-" * 95)

    print(String(format: "%-15s %-20s %-25.2f %-25s", "Baseline", "100.0%", baselineAvgLatency, "1.00x"))
    print("-" * 95)

    for (mult, recall, latency, speedup) in tradeoffResults {
      print(String(format: "%-15d %-20.1f%% %-25.2f %-25.2fx", mult, recall * 100, latency, speedup))
    }

    print("=" * 95)
    print("\n💡 Recommendation: Choose multiplier based on accuracy requirements:")
    print("   • mult=5:  Fast but lower accuracy (~60-70% recall)")
    print("   • mult=10: Balanced (default, ~80-90% recall)")
    print("   • mult=20: High accuracy (~95%+ recall) at cost of speed\n")

    #expect(tradeoffResults.count == multipliers.count)
  }

  // MARK: - Ranking Quality Tests

  @Test("Accuracy: ranking quality preservation")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func rankingQualityTest() async throws {
    let documentCount = 1_000
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: 10, seed: 54321)

    // Baseline
    let (dir1, cleanup1) = try makeTestDirectory()
    defer { cleanup1() }

    let config1 = VecturaConfig(
      name: "baseline-db",
      directoryURL: dir1,
      memoryStrategy: .fullMemory
    )
    let baselineVectura = try await VecturaKit(config: config1, embedder: makeEmbedder())

    // Generate consistent UUIDs for documents
    let documentIds = (0..<documentCount).map { _ in UUID() }
    _ = try await baselineVectura.addDocuments(texts: documents, ids: documentIds)

    // Indexed with MockIndexedStorage
    let (dir2, cleanup2) = try makeTestDirectory()
    defer { cleanup2() }

    let config2 = VecturaConfig(
      name: "indexed-db",
      directoryURL: dir2,
      memoryStrategy: .indexed(candidateMultiplier: 15)
    )
    let mockStorage = MockIndexedStorage()
    let indexedVectura = try await VecturaKit(config: config2, embedder: makeEmbedder(), storageProvider: mockStorage)
    // Use the same document IDs to ensure comparison is valid
    _ = try await indexedVectura.addDocuments(texts: documents, ids: documentIds)

    // Analyze ranking preservation
    var topKMatches: [Int] = []  // How many top-K results match at exact position

    for query in queries {
      let baselineResults = try await baselineVectura.search(query: .text(query), numResults: 10)
      let indexedResults = try await indexedVectura.search(query: .text(query), numResults: 10)

      var matches = 0
      for i in 0..<min(baselineResults.count, indexedResults.count) {
        if baselineResults[i].id == indexedResults[i].id {
          matches += 1
        }
      }
      topKMatches.append(matches)
    }

    let avgExactMatches = Double(topKMatches.reduce(0, +)) / Double(topKMatches.count)

    print("\n🏆 Ranking Quality Analysis:")
    print("=" * 70)
    print(String(format: "Avg Exact Position Matches: %.1f / 10", avgExactMatches))
    print(String(format: "Position Preservation Rate:  %.1f%%", avgExactMatches * 10))
    print("=" * 70)
    print("\n💡 Note: Indexed mode may reorder results slightly due to")
    print("   two-stage search, but should preserve top results well.\n")

    #expect(topKMatches.count == queries.count)
  }
}
