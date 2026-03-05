import Foundation
import Testing
@testable import VecturaKit

/// Harder, more realistic performance workloads.
///
/// These tests are opt-in and intended for stress/regression analysis:
/// - cold vs warm query behavior
/// - queueing pressure from multi-client access
/// - mixed read/write traffic
///
/// Enable with:
/// `VECTURA_PERF_PROFILE=realistic swift test --filter RealisticWorkloadSuite`
@Suite("Realistic Workload Benchmarks", .serialized)
struct RealisticWorkloadSuite {
  private struct RealisticReport {
    let strategyLabel: String
    let documentCount: Int
    let queryCount: Int
    let concurrentClients: Int
    let mixedOperationCount: Int
    let ingestionDocsPerSecond: Double
    let coldStartLatencies: [UInt64]
    let warmLatencies: [UInt64]
    let concurrentLatencies: [UInt64]
    let mixedSearchLatencies: [UInt64]
    let mixedWriteLatencies: [UInt64]

    var coldP95Ms: Double { percentile(0.95, coldStartLatencies) }
    var warmAvgMs: Double { average(warmLatencies) }
    var warmP99Ms: Double { percentile(0.99, warmLatencies) }
    var warmP999Ms: Double { percentile(0.999, warmLatencies) }
    var concurrentP99Ms: Double { percentile(0.99, concurrentLatencies) }
    var mixedSearchP99Ms: Double { percentile(0.99, mixedSearchLatencies) }
    var mixedWriteP95Ms: Double { percentile(0.95, mixedWriteLatencies) }

    func queriesPerSecond(for latencies: [UInt64]) -> Double {
      guard !latencies.isEmpty else { return 0 }
      let totalSeconds = Double(latencies.reduce(0, +)) / 1_000_000_000.0
      guard totalSeconds > 0 else { return 0 }
      return Double(latencies.count) / totalSeconds
    }

    var warmQueriesPerSecond: Double { queriesPerSecond(for: warmLatencies) }
    var concurrentQueriesPerSecond: Double { queriesPerSecond(for: concurrentLatencies) }

    private func average(_ values: [UInt64]) -> Double {
      guard !values.isEmpty else { return 0 }
      return Double(values.reduce(0, +)) / Double(values.count) / 1_000_000.0
    }

    private func percentile(_ percentileValue: Double, _ values: [UInt64]) -> Double {
      guard !values.isEmpty else { return 0 }
      let sorted = values.sorted()
      let index = min(Int(Double(sorted.count) * percentileValue), sorted.count - 1)
      return Double(sorted[index]) / 1_000_000.0
    }
  }

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("RealisticWorkloadSuite-\(UUID().uuidString)", isDirectory: true)
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

  private func shouldRunRealisticSuite(testName: String) -> Bool {
    guard PerformanceTestConfig.runRealisticBenchmarks else {
      print(
        "Skipping \(testName). Set VECTURA_PERF_PROFILE=realistic (or VECTURA_PERF_REALISTIC=1) to enable."
      )
      return false
    }
    return true
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func createVectura(
    directory: URL,
    databaseName: String,
    strategy: VecturaConfig.MemoryStrategy
  ) async throws -> VecturaKit {
    let config = try VecturaConfig(
      name: databaseName,
      directoryURL: directory,
      memoryStrategy: strategy
    )
    return try await VecturaKit(
      config: config,
      embedder: PerformanceTestConfig.makeEmbedder()
    )
  }

  private func elapsedNs(since start: UInt64) -> UInt64 {
    DispatchTime.now().uptimeNanoseconds - start
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func measureRealisticWorkload(
    strategy: VecturaConfig.MemoryStrategy,
    strategyLabel: String
  ) async throws -> RealisticReport {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()
    let documentCount = PerformanceTestConfig.realisticDocumentCount
    let queryCount = PerformanceTestConfig.realisticQueryCount
    let concurrentClients = PerformanceTestConfig.realisticConcurrentClients
    let coldRuns = PerformanceTestConfig.realisticColdRuns
    let mixedOperationCount = PerformanceTestConfig.realisticMixedOperationCount

    let documents = generator.generateRealisticCorpus(
      count: documentCount,
      seed: 20260305
    )
    let queries = generator.generateRealisticQueries(
      count: max(queryCount, coldRuns * concurrentClients),
      seed: 20260306
    )
    let mixedTexts = generator.generateRealisticCorpus(
      count: mixedOperationCount + 512,
      minWords: 12,
      maxWords: 120,
      duplicateRate: 0.05,
      seed: 20260307
    )

    let dbName = "realistic-db-\(strategyLabel)"
    let vectura = try await createVectura(
      directory: directory,
      databaseName: dbName,
      strategy: strategy
    )

    let ingestionStart = DispatchTime.now().uptimeNanoseconds
    let initialIds = try await vectura.addDocuments(texts: documents)
    let ingestionElapsed = elapsedNs(since: ingestionStart)
    let ingestionDocsPerSecond = Double(documentCount) / (Double(ingestionElapsed) / 1_000_000_000.0)

    var coldStartLatencies: [UInt64] = []
    coldStartLatencies.reserveCapacity(coldRuns)
    for i in 0..<coldRuns {
      let start = DispatchTime.now().uptimeNanoseconds
      let coldVectura = try await createVectura(
        directory: directory,
        databaseName: dbName,
        strategy: strategy
      )
      _ = try await coldVectura.search(query: .text(queries[i % queries.count]), numResults: 10)
      coldStartLatencies.append(elapsedNs(since: start))
    }

    for i in 0..<10 {
      _ = try await vectura.search(query: .text(queries[i % queries.count]), numResults: 10)
    }

    var warmLatencies: [UInt64] = []
    warmLatencies.reserveCapacity(queryCount)
    for i in 0..<queryCount {
      let start = DispatchTime.now().uptimeNanoseconds
      _ = try await vectura.search(query: .text(queries[i % queries.count]), numResults: 10)
      warmLatencies.append(elapsedNs(since: start))
    }

    let perClientQueryCount = max(1, queryCount / max(1, concurrentClients))
    let concurrentLatencies = try await withThrowingTaskGroup(of: [UInt64].self) { group in
      for client in 0..<concurrentClients {
        group.addTask {
          var latencies: [UInt64] = []
          latencies.reserveCapacity(perClientQueryCount)
          for i in 0..<perClientQueryCount {
            let queryIndex = (client * perClientQueryCount + i) % queries.count
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try await vectura.search(query: .text(queries[queryIndex]), numResults: 10)
            latencies.append(self.elapsedNs(since: start))
          }
          return latencies
        }
      }

      var flattened: [UInt64] = []
      for try await latencies in group {
        flattened.append(contentsOf: latencies)
      }
      return flattened
    }

    var mutableIds = initialIds
    var mixedSearchLatencies: [UInt64] = []
    var mixedWriteLatencies: [UInt64] = []
    mixedSearchLatencies.reserveCapacity(mixedOperationCount)
    mixedWriteLatencies.reserveCapacity(mixedOperationCount / 3)

    var operationCursor = 0
    for i in 0..<mixedOperationCount {
      let mode = i % 100
      if mode < 70 {
        let start = DispatchTime.now().uptimeNanoseconds
        let query = queries[(operationCursor + i) % queries.count]
        _ = try await vectura.search(query: .text(query), numResults: 10)
        mixedSearchLatencies.append(elapsedNs(since: start))
      } else if mode < 90 {
        let start = DispatchTime.now().uptimeNanoseconds
        let text = mixedTexts[(operationCursor + i) % mixedTexts.count]
        let id = try await vectura.addDocument(text: text)
        mutableIds.append(id)
        mixedWriteLatencies.append(elapsedNs(since: start))
      } else if !mutableIds.isEmpty {
        let start = DispatchTime.now().uptimeNanoseconds
        let targetIndex = (operationCursor + i) % mutableIds.count
        let id = mutableIds[targetIndex]
        let updatedText = mixedTexts[(operationCursor + i * 3 + 1) % mixedTexts.count]
        try await vectura.updateDocument(id: id, newText: updatedText)
        mixedWriteLatencies.append(elapsedNs(since: start))
      }
      operationCursor += 1
    }

    return RealisticReport(
      strategyLabel: strategyLabel,
      documentCount: documentCount,
      queryCount: queryCount,
      concurrentClients: concurrentClients,
      mixedOperationCount: mixedOperationCount,
      ingestionDocsPerSecond: ingestionDocsPerSecond,
      coldStartLatencies: coldStartLatencies,
      warmLatencies: warmLatencies,
      concurrentLatencies: concurrentLatencies,
      mixedSearchLatencies: mixedSearchLatencies,
      mixedWriteLatencies: mixedWriteLatencies
    )
  }

  private func printReport(_ report: RealisticReport) {
    print("\n" + "=" * 84)
    print("Realistic Workload Report: \(report.strategyLabel)")
    print("=" * 84)
    print("Dataset: \(report.documentCount) docs")
    print("Queries: \(report.queryCount) warm, \(report.concurrentClients) concurrent clients")
    print("Mixed Ops: \(report.mixedOperationCount) (70% search / 30% writes)")
    print("")
    print(String(format: "%-36@ %.2f", "Ingestion Throughput (docs/sec):", report.ingestionDocsPerSecond))
    print(String(format: "%-36@ %.2f ms", "Cold Start P95:", report.coldP95Ms))
    print(String(format: "%-36@ %.2f ms", "Warm Avg:", report.warmAvgMs))
    print(String(format: "%-36@ %.2f ms", "Warm P99:", report.warmP99Ms))
    print(String(format: "%-36@ %.2f ms", "Warm P99.9:", report.warmP999Ms))
    print(String(format: "%-36@ %.2f", "Warm QPS:", report.warmQueriesPerSecond))
    print(String(format: "%-36@ %.2f ms", "Concurrent P99:", report.concurrentP99Ms))
    print(String(format: "%-36@ %.2f", "Concurrent QPS:", report.concurrentQueriesPerSecond))
    print(String(format: "%-36@ %.2f ms", "Mixed Search P99:", report.mixedSearchP99Ms))
    print(String(format: "%-36@ %.2f ms", "Mixed Write P95:", report.mixedWriteP95Ms))
    print("=" * 84 + "\n")
  }

  @Test("Realistic: fullMemory cold/warm and mixed workload (opt-in)")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func realisticFullMemoryProfile() async throws {
    guard shouldRunRealisticSuite(testName: "realisticFullMemoryProfile") else { return }

    let report = try await measureRealisticWorkload(
      strategy: .fullMemory,
      strategyLabel: "fullMemory"
    )
    printReport(report)

    #expect(report.documentCount == PerformanceTestConfig.realisticDocumentCount)
    #expect(!report.warmLatencies.isEmpty)
    #expect(!report.concurrentLatencies.isEmpty)
    #expect(!report.mixedSearchLatencies.isEmpty)
    #expect(report.warmP999Ms >= report.warmP99Ms)
  }

  @Test("Realistic: indexed cold/warm and mixed workload (opt-in)")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func realisticIndexedProfile() async throws {
    guard shouldRunRealisticSuite(testName: "realisticIndexedProfile") else { return }

    let report = try await measureRealisticWorkload(
      strategy: .indexed(),
      strategyLabel: "indexed"
    )
    printReport(report)

    #expect(report.documentCount == PerformanceTestConfig.realisticDocumentCount)
    #expect(!report.warmLatencies.isEmpty)
    #expect(!report.concurrentLatencies.isEmpty)
    #expect(!report.mixedWriteLatencies.isEmpty)
    #expect(report.warmP999Ms >= report.warmP99Ms)
  }
}
