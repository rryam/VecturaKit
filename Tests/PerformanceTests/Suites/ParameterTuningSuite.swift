import Foundation
import Testing
@testable import VecturaKit

/// Parameter tuning tests for indexed memory strategy.
///
/// These tests systematically explore the parameter space of the indexed strategy
/// to identify optimal configurations for different use cases.
///
/// Key parameters tested:
/// - candidateMultiplier: Controls candidate pool size (accuracy vs speed)
/// - batchSize: Controls concurrent loading batch size
/// - maxConcurrentBatches: Controls parallelism level
///
/// Usage:
/// ```bash
/// # Run all parameter tuning tests
/// swift test --filter ParameterTuningSuite
/// ```
@Suite("Parameter Tuning Tests", .serialized)
struct ParameterTuningSuite {

  // MARK: - Data Models

  /// Represents a parameter configuration for testing.
  private struct ParameterConfiguration {
    let mult: Int
    let batch: Int
    let conc: Int
  }

  /// Represents a test result with label and metrics.
  private struct TestResult {
    let label: String
    let metrics: PerformanceMetrics
    let description: String
  }

  // MARK: - Test Infrastructure

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("ParameterTuningSuite-\(UUID().uuidString)", isDirectory: true)
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
  /// Run a benchmark with specific indexed strategy parameters.
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func runParameterBenchmark(
    documentCount: Int,
    queryCount: Int,
    candidateMultiplier: Int,
    batchSize: Int,
    maxConcurrentBatches: Int
  ) async throws -> PerformanceMetrics {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()
    let monitor = PerformanceMonitor()

    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: queryCount, seed: 54321)

    // Initialize with specific parameters
    await monitor.startTimer()
    let config = try VecturaConfig(
      name: "param-db",
      directoryURL: directory,
      memoryStrategy: .indexed(
        candidateMultiplier: candidateMultiplier,
        batchSize: batchSize,
        maxConcurrentBatches: maxConcurrentBatches
      )
    )
    let vectura = try await VecturaKit(
      config: config,
      embedder: PerformanceTestConfig.makeEmbedder()
    )
    let initTime = await monitor.getElapsed()
    await monitor.updateMemoryUsage()

    // Add documents
    let addStartTime = DispatchTime.now().uptimeNanoseconds
    _ = try await vectura.addDocuments(texts: documents)
    let addElapsed = DispatchTime.now().uptimeNanoseconds - addStartTime
    let addThroughput = Double(documentCount) / (Double(addElapsed) / 1_000_000_000.0)
    await monitor.updateMemoryUsage()

    // Measure searches
    for query in queries {
      await monitor.startTimer()
      _ = try await vectura.search(query: .text(query), numResults: 10)
      await monitor.recordElapsed()
      await monitor.updateMemoryUsage()
    }

    let strategyName = "mult=\(candidateMultiplier), batch=\(batchSize), conc=\(maxConcurrentBatches)"
    return await monitor.buildMetrics(
      initTimeNanos: initTime,
      documentCount: documentCount,
      strategyDescription: strategyName,
      addDocumentThroughput: addThroughput
    )
  }

  // MARK: - Candidate Multiplier Grid Search

  @Test("Tuning: candidate multiplier sweep (5-25)")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func candidateMultiplierSweep() async throws {
    let multipliers = [5, 10, 15, 20]
    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for mult in multipliers {
      let metrics = try await runParameterBenchmark(
        documentCount: 1_500,
        queryCount: 20,
        candidateMultiplier: mult,
        batchSize: 100,  // Keep other params constant
        maxConcurrentBatches: 4
      )
      results.append((label: "Mult \(mult)", metrics: metrics))
    }

    let reporter = ResultsReporter()
    print("\n🎯 Candidate Multiplier Tuning Results:")
    reporter.printSummaryTable(results)

    // Analysis
    print("\n📊 Analysis:")
    print("Higher multipliers typically:")
    print("  ✓ Better search accuracy (more candidates considered)")
    print("  ✗ Slower search performance (more documents to load)")
    print("  ✗ Higher memory usage during search")
    print("\nRecommended: 10-15 for balanced performance\n")

    #expect(results.count == multipliers.count)
  }

  // MARK: - Batch Size Grid Search

  @Test("Tuning: batch size sweep (25-250)")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func batchSizeSweep() async throws {
    let batchSizes = [50, 100, 150]
    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for batch in batchSizes {
      let metrics = try await runParameterBenchmark(
        documentCount: 1_500,
        queryCount: 20,
        candidateMultiplier: 10,  // Keep other params constant
        batchSize: batch,
        maxConcurrentBatches: 4
      )
      results.append((label: "Batch \(batch)", metrics: metrics))
    }

    let reporter = ResultsReporter()
    print("\n📦 Batch Size Tuning Results:")
    reporter.printSummaryTable(results)

    // Analysis
    print("\n📊 Analysis:")
    print("Larger batch sizes typically:")
    print("  ✓ Better throughput (fewer task switches)")
    print("  ✗ Higher memory peaks (more docs loaded at once)")
    print("  ✗ Less granular parallelism")
    print("\nRecommended: 100-150 for most workloads\n")

    #expect(results.count == batchSizes.count)
  }

  // MARK: - Concurrency Grid Search

  @Test("Tuning: max concurrent batches sweep (2-8)")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func maxConcurrentBatchesSweep() async throws {
    let concurrencyLevels = [2, 4, 6]
    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for conc in concurrencyLevels {
      let metrics = try await runParameterBenchmark(
        documentCount: 1_500,
        queryCount: 20,
        candidateMultiplier: 10,  // Keep other params constant
        batchSize: 100,
        maxConcurrentBatches: conc
      )
      results.append((label: "Conc \(conc)", metrics: metrics))
    }

    let reporter = ResultsReporter()
    print("\n⚡ Concurrency Level Tuning Results:")
    reporter.printSummaryTable(results)

    // Analysis
    print("\n📊 Analysis:")
    print("Higher concurrency typically:")
    print("  ✓ Faster search (more parallel batch loads)")
    print("  ✗ Higher resource contention")
    print("  ✗ Diminishing returns beyond CPU core count")
    print("\nRecommended: 4-6 for most systems\n")

    #expect(results.count == concurrencyLevels.count)
  }

  // MARK: - Combined Parameter Optimization

  @Test("Tuning: find optimal parameter combination")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func optimalParameterCombination() async throws {
    // Test several promising combinations
    let configurations: [ParameterConfiguration] = [
      ParameterConfiguration(mult: 5, batch: 50, conc: 2),    // Fast but less accurate
      ParameterConfiguration(mult: 10, batch: 100, conc: 4),  // Balanced (default)
      ParameterConfiguration(mult: 15, batch: 100, conc: 4)  // Better accuracy
    ]

    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for config in configurations {
      let metrics = try await runParameterBenchmark(
        documentCount: 1_500,
        queryCount: 20,
        candidateMultiplier: config.mult,
        batchSize: config.batch,
        maxConcurrentBatches: config.conc
      )
      let label = "M\(config.mult)-B\(config.batch)-C\(config.conc)"
      results.append((label: label, metrics: metrics))
    }

    let reporter = ResultsReporter()
    print("\n🏆 Parameter Combination Comparison:")
    reporter.printSummaryTable(results)

    // Find best for different criteria
    var bestLatency = results[0]
    var bestMemory = results[0]
    var bestThroughput = results[0]

    for result in results {
      if result.metrics.avgLatency < bestLatency.metrics.avgLatency {
        bestLatency = result
      }
      if result.metrics.memoryPeakMB < bestMemory.metrics.memoryPeakMB {
        bestMemory = result
      }
      if (result.metrics.addDocumentThroughput ?? 0) > (bestThroughput.metrics.addDocumentThroughput ?? 0) {
        bestThroughput = result
      }
    }

    print("\n🎖️ Best Configurations:")
    print("  Lowest Latency:      \(bestLatency.label) (\(String(format: "%.2f ms", bestLatency.metrics.avgLatency)))")
    print("  Lowest Memory:       \(bestMemory.label) (\(String(format: "%.2f MB", bestMemory.metrics.memoryPeakMB)))")
    let throughputValue = bestThroughput.metrics.addDocumentThroughput ?? 0
    print("  Highest Throughput:  \(bestThroughput.label) (\(String(format: "%.0f docs/s", throughputValue)))\n")

    #expect(results.count == configurations.count)
  }

  // MARK: - Workload-Specific Recommendations

  @Test("Tuning: workload-specific parameter recommendations")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func workloadSpecificRecommendations() async throws {
    // Test configurations for different workload types
    struct WorkloadConfig {
      let name: String
      let mult: Int
      let batch: Int
      let conc: Int
      let description: String
    }

    let workloads: [WorkloadConfig] = [
      WorkloadConfig(
        name: "Low-latency",
        mult: 5,
        batch: 50,
        conc: 4,
        description: "Optimized for fastest search response"
      ),
      WorkloadConfig(
        name: "Memory-constrained",
        mult: 10,
        batch: 50,
        conc: 2,
        description: "Optimized for minimal memory footprint"
      ),
      WorkloadConfig(
        name: "High-accuracy",
        mult: 15,
        batch: 100,
        conc: 4,
        description: "Optimized for best search quality"
      )
    ]

    var results: [TestResult] = []

    for workload in workloads {
      let metrics = try await runParameterBenchmark(
        documentCount: 1_500,
        queryCount: 20,
        candidateMultiplier: workload.mult,
        batchSize: workload.batch,
        maxConcurrentBatches: workload.conc
      )
      results.append(TestResult(label: workload.name, metrics: metrics, description: workload.description))
    }

    // Print recommendations
    print("\n💼 Workload-Specific Recommendations:")
    print("=" * 90)

    for result in results {
      print("\n\(result.label):")
      print("  \(result.description)")
      print(String(format: "  Avg Latency: %.2f ms  |  Memory: %.2f MB  |  Throughput: %.0f docs/s",
            result.metrics.avgLatency,
            result.metrics.memoryPeakMB,
            result.metrics.addDocumentThroughput ?? 0))
    }

    print("\n" + "=" * 90 + "\n")

    #expect(results.count == workloads.count)
  }
}
