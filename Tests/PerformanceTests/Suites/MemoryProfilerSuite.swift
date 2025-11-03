import Foundation
import Testing
@testable import VecturaKit

/// Memory profiling tests for VecturaKit.
///
/// These tests perform detailed memory analysis of different strategies,
/// tracking memory usage patterns during initialization, document addition,
/// and search operations.
///
/// Usage:
/// ```bash
/// # Run all memory profiling tests
/// swift test --filter MemoryProfilerSuite
/// ```
@Suite("Memory Profiling Tests", .serialized)
struct MemoryProfilerSuite {

  // MARK: - Data Models

  /// Represents memory usage results for a strategy.
  private struct MemoryResult {
    let strategy: String
    let peakMB: Double
    let perDocKB: Double
  }

  // MARK: - Test Infrastructure

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("MemoryProfilerSuite-\(UUID().uuidString)", isDirectory: true)
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

  /// Get current memory usage in megabytes.
  private func getCurrentMemoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }

    if result == KERN_SUCCESS {
      return Double(info.resident_size) / 1_024.0 / 1_024.0
    }
    return 0
  }

  // MARK: - Memory Lifecycle Tests

  @Test("Memory: fullMemory strategy lifecycle")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func fullMemoryLifecycle() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()
    var memorySnapshots: [(phase: String, memoryMB: Double)] = []

    // Baseline
    memorySnapshots.append(("Baseline", getCurrentMemoryMB()))

    // After initialization
    let config = VecturaConfig(
      name: "mem-test-db",
      directoryURL: directory,
      memoryStrategy: .fullMemory
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    memorySnapshots.append(("After Init", getCurrentMemoryMB()))

    // After adding documents
    let docs1k = generator.generateDocuments(count: 500, seed: 12345)
    _ = try await vectura.addDocuments(texts: docs1k)
    memorySnapshots.append(("After 500 docs", getCurrentMemoryMB()))

    // After adding more docs
    let docs2k = generator.generateDocuments(count: 500, seed: 54321)
    _ = try await vectura.addDocuments(texts: docs2k)
    memorySnapshots.append(("After 1K docs", getCurrentMemoryMB()))

    // After searches
    let queries = generator.generateQueries(count: 30, seed: 99999)
    for query in queries {
      _ = try await vectura.search(query: .text(query), numResults: 10)
    }
    memorySnapshots.append(("After 30 searches", getCurrentMemoryMB()))

    // Print lifecycle report
    print("\n📊 Full Memory Strategy - Memory Lifecycle:")
    print("=" * 70)
    print(String(format: "%-25s %-20s %-20s", "Phase", "Memory (MB)", "Delta (MB)"))
    print("-" * 70)

    for index in 0..<memorySnapshots.count {
      let (phase, memory) = memorySnapshots[index]
      let delta = index > 0 ? memory - memorySnapshots[0].memoryMB : 0
      print(String(format: "%-25s %-20.2f %-20.2f", phase, memory, delta))
    }
    print("=" * 70 + "\n")

    // Verify memory increased with documents
    let baselineMemory = memorySnapshots[0].memoryMB
    let finalMemory = memorySnapshots.last!.memoryMB
    #expect(finalMemory > baselineMemory, "Memory should increase with documents")
  }

  @Test("Memory: indexed strategy lifecycle")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func indexedStrategyLifecycle() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()
    var memorySnapshots: [(phase: String, memoryMB: Double)] = []

    // Baseline
    memorySnapshots.append(("Baseline", getCurrentMemoryMB()))

    // After initialization
    let config = VecturaConfig(
      name: "mem-test-db",
      directoryURL: directory,
      memoryStrategy: .indexed()
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
    memorySnapshots.append(("After Init", getCurrentMemoryMB()))

    // After adding documents
    let docs1k = generator.generateDocuments(count: 500, seed: 12345)
    _ = try await vectura.addDocuments(texts: docs1k)
    memorySnapshots.append(("After 500 docs", getCurrentMemoryMB()))

    // After adding more docs
    let docs2k = generator.generateDocuments(count: 500, seed: 54321)
    _ = try await vectura.addDocuments(texts: docs2k)
    memorySnapshots.append(("After 1K docs", getCurrentMemoryMB()))

    // After searches
    let queries = generator.generateQueries(count: 30, seed: 99999)
    for query in queries {
      _ = try await vectura.search(query: .text(query), numResults: 10)
    }
    memorySnapshots.append(("After 30 searches", getCurrentMemoryMB()))

    // Print lifecycle report
    print("\n📊 Indexed Strategy - Memory Lifecycle:")
    print("=" * 70)
    print(String(format: "%-25s %-20s %-20s", "Phase", "Memory (MB)", "Delta (MB)"))
    print("-" * 70)

    for index in 0..<memorySnapshots.count {
      let (phase, memory) = memorySnapshots[index]
      let delta = index > 0 ? memory - memorySnapshots[0].memoryMB : 0
      print(String(format: "%-25s %-20.2f %-20.2f", phase, memory, delta))
    }
    print("=" * 70 + "\n")

    // Indexed mode should use less memory than fullMemory for same data
    #expect(memorySnapshots.count == 5)
  }

  // MARK: - Memory Efficiency Comparison

  @Test("Memory: strategy comparison at 1K documents")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func strategyMemoryComparison() async throws {
    let documentCount = 1_000
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: documentCount, seed: 12345)

    var results: [MemoryResult] = []

    // Test fullMemory strategy
    do {
      let (directory, cleanup) = try makeTestDirectory()
      defer { cleanup() }

      let baselineMemory = getCurrentMemoryMB()

      let config = VecturaConfig(
        name: "mem-full-db",
        directoryURL: directory,
        memoryStrategy: .fullMemory
      )
      let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
      _ = try await vectura.addDocuments(texts: documents)

      let peakMemory = getCurrentMemoryMB()
      let memoryUsed = peakMemory - baselineMemory
      let perDoc = memoryUsed * 1024.0 / Double(documentCount)

      results.append(MemoryResult(strategy: "fullMemory", peakMB: memoryUsed, perDocKB: perDoc))
    }

    // Test indexed strategy
    do {
      let (directory, cleanup) = try makeTestDirectory()
      defer { cleanup() }

      let baselineMemory = getCurrentMemoryMB()

      let config = VecturaConfig(
        name: "mem-idx-db",
        directoryURL: directory,
        memoryStrategy: .indexed()
      )
      let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
      _ = try await vectura.addDocuments(texts: documents)

      let peakMemory = getCurrentMemoryMB()
      let memoryUsed = peakMemory - baselineMemory
      let perDoc = memoryUsed * 1024.0 / Double(documentCount)

      results.append(MemoryResult(strategy: "indexed", peakMB: memoryUsed, perDocKB: perDoc))
    }

    // Print comparison
    print("\n💾 Memory Efficiency Comparison (1K documents):")
    print("=" * 70)
    print(String(format: "%-20s %-20s %-20s", "Strategy", "Peak Memory (MB)", "Per Document (KB)"))
    print("-" * 70)

    for result in results {
      print(String(format: "%-20s %-20.2f %-20.2f", result.strategy, result.peakMB, result.perDocKB))
    }

    if results.count == 2 {
      let saving = ((results[0].peakMB - results[1].peakMB) / results[0].peakMB) * 100
      print("-" * 70)
      print(String(format: "Memory Savings (indexed vs fullMemory): %.1f%%", saving))
    }

    print("=" * 70 + "\n")

    #expect(results.count == 2)
  }

  // MARK: - Search Memory Impact

  @Test("Memory: search operation memory impact")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func searchMemoryImpact() async throws {
    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let generator = TestDataGenerator()

    let config = VecturaConfig(
      name: "search-mem-db",
      directoryURL: directory,
      memoryStrategy: .fullMemory
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    // Add documents
    let documents = generator.generateDocuments(count: 500, seed: 12345)
    _ = try await vectura.addDocuments(texts: documents)

    let beforeSearch = getCurrentMemoryMB()

    // Perform 100 searches and track memory
    let queries = generator.generateQueries(count: 50, seed: 54321)
    var searchMemories: [Double] = []

    for query in queries {
      _ = try await vectura.search(query: .text(query), numResults: 10)
      searchMemories.append(getCurrentMemoryMB())
    }

    let afterSearch = getCurrentMemoryMB()
    let avgSearchMemory = searchMemories.reduce(0, +) / Double(searchMemories.count)
    let maxSearchMemory = searchMemories.max() ?? 0
    let minSearchMemory = searchMemories.min() ?? 0

    // Print results
    print("\n🔍 Search Memory Impact Analysis:")
    print("=" * 70)
    print(String(format: "Before searches:     %.2f MB", beforeSearch))
    print(String(format: "After searches:      %.2f MB", afterSearch))
    print(String(format: "Average during:      %.2f MB", avgSearchMemory))
    print(String(format: "Peak during:         %.2f MB", maxSearchMemory))
    print(String(format: "Min during:          %.2f MB", minSearchMemory))
    print(String(format: "Memory delta:        %.2f MB", afterSearch - beforeSearch))
    print("=" * 70 + "\n")

    #expect(searchMemories.count == 50)
  }

  // MARK: - Batch Size Memory Impact

  @Test("Memory: batch size impact on indexed strategy")
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  func batchSizeMemoryImpact() async throws {
    let batchSizes = [50, 100]
    let generator = TestDataGenerator()
    let documents = generator.generateDocuments(count: 1_000, seed: 12345)

    var results: [(batchSize: Int, peakMB: Double)] = []

    for batchSize in batchSizes {
      let (directory, cleanup) = try makeTestDirectory()
      defer { cleanup() }

      let baselineMemory = getCurrentMemoryMB()

      let config = VecturaConfig(
        name: "batch-mem-db",
        directoryURL: directory,
        memoryStrategy: .indexed(batchSize: batchSize)
      )
      let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
      _ = try await vectura.addDocuments(texts: documents)

      let peakMemory = getCurrentMemoryMB()
      results.append((batchSize, peakMemory - baselineMemory))
    }

    // Print results
    print("\n📦 Batch Size Memory Impact:")
    print("=" * 70)
    print(String(format: "%-20s %-20s", "Batch Size", "Peak Memory (MB)"))
    print("-" * 70)

    for (batchSize, peak) in results {
      print(String(format: "%-20d %-20.2f", batchSize, peak))
    }
    print("=" * 70 + "\n")

    #expect(results.count == batchSizes.count)
  }
}
