import Foundation
import Testing
@testable import VecturaKit

/// Scalability tests for VecturaKit memory strategies.
///
/// These tests measure how well different strategies scale from small to large datasets,
/// examining the growth curves for search latency, memory usage, and throughput.
///
/// Usage:
/// ```bash
/// # Run all scalability tests
/// swift test --filter ScalabilitySuite
///
/// # Run specific scale test
/// swift test --filter ScalabilitySuite.fullMemoryScaling
/// ```
@Suite("Scalability Tests", .serialized)
struct ScalabilitySuite {

    // MARK: - Test Infrastructure

    private func makeTestDirectory() throws -> (URL, () -> Void) {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appendingPathComponent("ScalabilitySuite-\(UUID().uuidString)", isDirectory: true)
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

    /// Run a benchmark for a specific document count and strategy.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    private func runScaleBenchmark(
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
        let config = VecturaConfig(
            name: "scale-db",
            directoryURL: directory,
            memoryStrategy: strategy
        )
        let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())
        let initTime = await monitor.getElapsed()
        await monitor.updateMemoryUsage()

        // Measure document addition
        let addStartTime = DispatchTime.now().uptimeNanoseconds
        _ = try await vectura.addDocuments(texts: documents)
        let addElapsed = DispatchTime.now().uptimeNanoseconds - addStartTime
        let addThroughput = Double(documentCount) / (Double(addElapsed) / 1_000_000_000.0)
        await monitor.updateMemoryUsage()

        // Measure search performance
        for query in queries {
            await monitor.startTimer()
            _ = try await vectura.search(query: query, numResults: 10)
            await monitor.recordElapsed()
            await monitor.updateMemoryUsage()
        }

        return await monitor.buildMetrics(
            initTimeNanos: initTime,
            documentCount: documentCount,
            strategyDescription: strategyName,
            addDocumentThroughput: addThroughput
        )
    }

    // MARK: - Full Memory Scaling Tests

    @Test("Scaling: fullMemory strategy across dataset sizes")
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func fullMemoryScaling() async throws {
        let dataSizes = [500, 1_000, 1_500]
        var results: [(label: String, metrics: PerformanceMetrics)] = []

        for size in dataSizes {
            let metrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 20,
                strategy: .fullMemory,
                strategyName: "\(size) docs - fullMemory"
            )
            results.append((label: "\(size) docs", metrics: metrics))
        }

        let reporter = ResultsReporter()
        reporter.printSummaryTable(results)

        // Verify all tests completed
        for (index, (_, metrics)) in results.enumerated() {
            #expect(metrics.documentCount == dataSizes[index])
        }
    }

    // MARK: - Indexed Strategy Scaling Tests

    @Test("Scaling: indexed strategy across dataset sizes")
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func indexedStrategyScaling() async throws {
        let dataSizes = [500, 1_000, 1_500]
        var results: [(label: String, metrics: PerformanceMetrics)] = []

        for size in dataSizes {
            let metrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 20,
                strategy: .indexed(),
                strategyName: "\(size) docs - indexed"
            )
            results.append((label: "\(size) docs", metrics: metrics))
        }

        let reporter = ResultsReporter()
        reporter.printSummaryTable(results)

        // Verify all tests completed
        for (index, (_, metrics)) in results.enumerated() {
            #expect(metrics.documentCount == dataSizes[index])
        }
    }

    // MARK: - Cross-Strategy Scaling Comparison

    @Test("Scaling: fullMemory vs indexed growth curves")
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func crossStrategyScaling() async throws {
        let dataSizes = [500, 1_000, 1_500]
        var fullMemoryResults: [(label: String, metrics: PerformanceMetrics)] = []
        var indexedResults: [(label: String, metrics: PerformanceMetrics)] = []

        for size in dataSizes {
            // Full memory
            let fullMemMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 20,
                strategy: .fullMemory,
                strategyName: "\(size) docs - fullMemory"
            )
            fullMemoryResults.append((label: "Full \(size)", metrics: fullMemMetrics))

            // Indexed
            let indexedMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 20,
                strategy: .indexed(),
                strategyName: "\(size) docs - indexed"
            )
            indexedResults.append((label: "Idx \(size)", metrics: indexedMetrics))
        }

        // Print combined results
        let reporter = ResultsReporter()
        print("\n🔍 Full Memory Strategy Scaling:")
        reporter.printSummaryTable(fullMemoryResults)

        print("\n🔍 Indexed Strategy Scaling:")
        reporter.printSummaryTable(indexedResults)

        // Verify all tests completed
        #expect(fullMemoryResults.count == dataSizes.count)
        #expect(indexedResults.count == dataSizes.count)
    }

    // MARK: - Throughput Scaling Tests

    @Test("Scaling: document addition throughput")
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func addDocumentThroughputScaling() async throws {
        let dataSizes = [500, 1_000]
        var fullMemoryResults: [(label: String, metrics: PerformanceMetrics)] = []
        var indexedResults: [(label: String, metrics: PerformanceMetrics)] = []

        for size in dataSizes {
            // Full memory throughput
            let fullMemMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 10,
                strategy: .fullMemory,
                strategyName: "\(size) docs - fullMemory"
            )
            fullMemoryResults.append((label: "Full \(size)", metrics: fullMemMetrics))

            // Indexed throughput
            let indexedMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 10,
                strategy: .indexed(),
                strategyName: "\(size) docs - indexed"
            )
            indexedResults.append((label: "Idx \(size)", metrics: indexedMetrics))
        }

        // Print throughput comparison
        print("\n📊 Document Addition Throughput Comparison:")
        print("=" * 80)
        print(String(format: "%-20s %-25s %-25s", "Dataset Size", "Full Memory (docs/sec)", "Indexed (docs/sec)"))
        print("-" * 80)

        for index in 0..<dataSizes.count {
            let size = dataSizes[index]
            let fullThroughput = fullMemoryResults[index].metrics.addDocumentThroughput ?? 0
            let idxThroughput = indexedResults[index].metrics.addDocumentThroughput ?? 0

            print(String(
                format: "%-20d %-25.0f %-25.0f",
                size,
                fullThroughput,
                idxThroughput
            ))
        }
        print("=" * 80 + "\n")

        // Verify all tests completed
        #expect(fullMemoryResults.count == dataSizes.count)
        #expect(indexedResults.count == dataSizes.count)
    }

    // MARK: - Memory Growth Tests

    @Test("Scaling: memory usage growth patterns")
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func memoryGrowthPatterns() async throws {
        let dataSizes = [500, 1_000, 1_500]
        var fullMemoryResults: [(label: String, metrics: PerformanceMetrics)] = []
        var indexedResults: [(label: String, metrics: PerformanceMetrics)] = []

        for size in dataSizes {
            // Full memory
            let fullMemMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 15,
                strategy: .fullMemory,
                strategyName: "\(size) docs - fullMemory"
            )
            fullMemoryResults.append((label: "Full \(size)", metrics: fullMemMetrics))

            // Indexed
            let indexedMetrics = try await runScaleBenchmark(
                documentCount: size,
                queryCount: 15,
                strategy: .indexed(),
                strategyName: "\(size) docs - indexed"
            )
            indexedResults.append((label: "Idx \(size)", metrics: indexedMetrics))
        }

        // Print memory growth analysis
        print("\n💾 Memory Usage Growth Analysis:")
        print("=" * 100)
        print(String(format: "%-12s %-20s %-20s %-20s %-20s",
                     "Size", "Full Peak (MB)", "Full Per Doc (KB)", "Indexed Peak (MB)", "Indexed Per Doc (KB)"))
        print("-" * 100)

        for index in 0..<dataSizes.count {
            let size = dataSizes[index]
            let fullMem = fullMemoryResults[index].metrics
            let idxMem = indexedResults[index].metrics

            print(String(
                format: "%-12d %-20.2f %-20.2f %-20.2f %-20.2f",
                size,
                fullMem.memoryPeakMB,
                fullMem.memoryPerDocKB,
                idxMem.memoryPeakMB,
                idxMem.memoryPerDocKB
            ))
        }
        print("=" * 100 + "\n")

        // Verify results
        #expect(fullMemoryResults.count == dataSizes.count)
        #expect(indexedResults.count == dataSizes.count)
    }
}

// MARK: - String Extension

private extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
