import Foundation
import Testing
@testable import VecturaKit

/// Performance benchmarks for VecturaKit memory strategies.
///
/// These tests measure the performance characteristics introduced in commit 7979154e,
/// specifically the memory management strategies (automatic, fullMemory, indexed).
///
/// Usage:
/// ```bash
/// # Run all performance tests
/// swift test --filter BenchmarkSuite
///
/// # Run specific benchmark
/// swift test --filter BenchmarkSuite.smallDatasetFullMemory
/// ```
@Suite("Performance Benchmarks", .serialized)
struct BenchmarkSuite {

  // MARK: - Test Infrastructure

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("BenchmarkSuite-\(UUID().uuidString)", isDirectory: true)
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

  /// Run a complete benchmark for a given configuration.
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func runBenchmark(
    documentCount: Int,
    queryCount: Int,
    strategy: VecturaConfig.MemoryStrategy,
    strategyName: String
  ) async throws -> PerformanceMetrics {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()
    let monitor = PerformanceMonitor()

    // Generate test data
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)
    let queries = generator.generateQueries(count: queryCount, seed: 54321)

    // Measure initialization
    await monitor.startTimer()
    let config = try VecturaConfig(
      name: "benchmark-db",
      directoryURL: directory,
      memoryStrategy: strategy
    )
    let vectura = try await VecturaKit(
      config: config,
      embedder: PerformanceTestConfig.makeEmbedder()
    )
    let initTime = await monitor.getElapsed()
    await monitor.updateMemoryUsage()

    // Measure document addition
    let addStartTime = DispatchTime.now().uptimeNanoseconds
    _ = try await vectura.addDocuments(texts: documents)
    let addElapsed = DispatchTime.now().uptimeNanoseconds - addStartTime
    let addThroughput = Double(documentCount) / (Double(addElapsed) / 1_000_000_000.0)
    await monitor.updateMemoryUsage()

    // Warm-up searches (not measured)
    for query in queries.prefix(2) {
      _ = try await vectura.search(query: .text(query), numResults: 10)
    }

    // Measure search performance
    for query in queries {
      await monitor.startTimer()
      _ = try await vectura.search(query: .text(query), numResults: 10)
      await monitor.recordElapsed()
      await monitor.updateMemoryUsage()
    }

    // Build metrics
    return await monitor.buildMetrics(
      initTimeNanos: initTime,
      documentCount: documentCount,
      strategyDescription: strategyName,
      addDocumentThroughput: addThroughput
    )
  }

  // MARK: - Small Dataset Benchmarks (1,000 documents)

  @Test("Benchmark: 1K docs, fullMemory strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func smallDatasetFullMemory() async throws {
    let metrics = try await runBenchmark(
      documentCount: 1_000,
      queryCount: 50,
      strategy: .fullMemory,
      strategyName: "1K docs - fullMemory"
    )

    let reporter = ResultsReporter()
    reporter.printReport(metrics)

    // Basic sanity checks
    #expect(metrics.avgLatency < 100.0, "Average search latency should be < 100ms for 1K docs")
    #expect(metrics.documentCount == 1_000)
  }

  @Test("Benchmark: 1K docs, automatic strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func smallDatasetAutomatic() async throws {
    let metrics = try await runBenchmark(
      documentCount: 1_000,
      queryCount: 50,
      strategy: .automatic(),
      strategyName: "1K docs - automatic"
    )

    let reporter = ResultsReporter()
    reporter.printReport(metrics)

    #expect(metrics.avgLatency < 100.0, "Average search latency should be < 100ms for 1K docs")
    #expect(metrics.documentCount == 1_000)
  }

  // MARK: - Medium Dataset Benchmarks (10,000 documents)

  @Test("Benchmark: 3K docs, fullMemory strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func mediumDatasetFullMemory() async throws {
    let metrics = try await runBenchmark(
      documentCount: 3_000,
      queryCount: 30,
      strategy: .fullMemory,
      strategyName: "3K docs - fullMemory"
    )

    let reporter = ResultsReporter()
    reporter.printReport(metrics)

    #expect(metrics.avgLatency < 200.0, "Average search latency should be < 200ms for 3K docs")
    #expect(metrics.documentCount == 3_000)
  }

  @Test("Benchmark: 3K docs, indexed strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func mediumDatasetIndexed() async throws {
    let metrics = try await runBenchmark(
      documentCount: 3_000,
      queryCount: 30,
      strategy: .indexed(),
      strategyName: "3K docs - indexed"
    )

    let reporter = ResultsReporter()
    reporter.printReport(metrics)

    #expect(metrics.avgLatency < 500.0, "Average search latency should be < 500ms for 3K docs")
    #expect(metrics.documentCount == 3_000)
  }

  // MARK: - Strategy Comparison Tests

  @Test("Compare: fullMemory vs indexed at 2K docs")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func compareStrategies10K() async throws {
    let fullMemoryMetrics = try await runBenchmark(
      documentCount: 2_000,
      queryCount: 30,
      strategy: .fullMemory,
      strategyName: "2K docs - fullMemory"
    )

    let indexedMetrics = try await runBenchmark(
      documentCount: 2_000,
      queryCount: 30,
      strategy: .indexed(),
      strategyName: "2K docs - indexed"
    )

    let reporter = ResultsReporter()
    reporter.printComparison(fullMemoryMetrics, vs: indexedMetrics)

    // Both strategies should complete successfully
    #expect(fullMemoryMetrics.documentCount == 2_000)
    #expect(indexedMetrics.documentCount == 2_000)
  }

  // MARK: - Parameter Tuning Tests

  @Test("Indexed strategy: candidate multiplier comparison")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func candidateMultiplierComparison() async throws {
    let multipliers = [5, 10, 15, 20]
    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for multiplier in multipliers {
      let metrics = try await runBenchmark(
        documentCount: 2_000,
        queryCount: 20,
        strategy: .indexed(candidateMultiplier: multiplier),
        strategyName: "2K docs - indexed (mult=\(multiplier))"
      )
      results.append((label: "Multiplier \(multiplier)", metrics: metrics))
    }

    let reporter = ResultsReporter()
    reporter.printSummaryTable(results)

    // All configurations should complete
    for (_, metrics) in results {
      #expect(metrics.documentCount == 2_000)
    }
  }

  @Test("Indexed strategy: batch size comparison")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func batchSizeComparison() async throws {
    let batchSizes = [50, 100, 150, 200]
    var results: [(label: String, metrics: PerformanceMetrics)] = []

    for batchSize in batchSizes {
      let metrics = try await runBenchmark(
        documentCount: 2_000,
        queryCount: 20,
        strategy: .indexed(batchSize: batchSize),
        strategyName: "2K docs - indexed (batch=\(batchSize))"
      )
      results.append((label: "Batch \(batchSize)", metrics: metrics))
    }

    let reporter = ResultsReporter()
    reporter.printSummaryTable(results)

    // All configurations should complete
    for (_, metrics) in results {
      #expect(metrics.documentCount == 2_000)
    }
  }
}
