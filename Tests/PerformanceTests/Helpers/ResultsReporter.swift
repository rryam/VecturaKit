import Foundation

/// Formats and reports performance test results.
public struct ResultsReporter: Sendable {

    public init() {}

    /// Print a formatted performance report to stdout.
    ///
    /// - Parameter metrics: The performance metrics to report
    public func printReport(_ metrics: PerformanceMetrics) {
        print("\n" + "=".repeating(count: 70))
        print("Performance Report: \(metrics.strategyDescription)")
        print("=".repeating(count: 70))

        // Dataset Info
        print("\nDataset:")
        print("  Documents: \(metrics.documentCount.formatted())")

        // Initialization
        print("\nInitialization:")
        print("  Time: \(String(format: "%.2f ms", metrics.initTimeMs))")

        // Search Performance
        print("\nSearch Latency:")
        print("  Average: \(String(format: "%.2f ms", metrics.avgLatency))")
        print("  P50:     \(String(format: "%.2f ms", metrics.p50Latency))")
        print("  P95:     \(String(format: "%.2f ms", metrics.p95Latency))")
        print("  P99:     \(String(format: "%.2f ms", metrics.p99Latency))")
        print("  Samples: \(metrics.searchLatencies.count)")

        // Memory Usage
        print("\nMemory Usage:")
        print("  Peak:     \(String(format: "%.2f MB", metrics.memoryPeakMB))")
        print("  Per Doc:  \(String(format: "%.2f KB", metrics.memoryPerDocKB))")

        // Optional Metrics
        if let throughput = metrics.addDocumentThroughput {
            print("\nThroughput:")
            print("  Add Documents: \(String(format: "%.0f docs/sec", throughput))")
        }

        if let accuracy = metrics.accuracy {
            print("\nAccuracy:")
            print("  vs Baseline: \(String(format: "%.1f%%", accuracy * 100))")
        }

        print("=".repeating(count: 70) + "\n")
    }

    /// Print a comparison between two sets of metrics.
    ///
    /// - Parameters:
    ///   - baseline: Baseline metrics for comparison
    ///   - comparison: Metrics to compare against baseline
    public func printComparison(_ baseline: PerformanceMetrics, vs comparison: PerformanceMetrics) {
        print("\n" + "=".repeating(count: 70))
        print("Performance Comparison")
        print("=".repeating(count: 70))

        print("\nBaseline:   \(baseline.strategyDescription)")
        print("Comparison: \(comparison.strategyDescription)")

        // Search Latency Comparison
        print("\nSearch Latency (Comparison / Baseline):")
        printMetric("Average", baseline: baseline.avgLatency, comparison: comparison.avgLatency, unit: "ms")
        printMetric("P50", baseline: baseline.p50Latency, comparison: comparison.p50Latency, unit: "ms")
        printMetric("P95", baseline: baseline.p95Latency, comparison: comparison.p95Latency, unit: "ms")
        printMetric("P99", baseline: baseline.p99Latency, comparison: comparison.p99Latency, unit: "ms")

        // Memory Comparison
        print("\nMemory Usage (Comparison / Baseline):")
        printMetric("Peak", baseline: baseline.memoryPeakMB, comparison: comparison.memoryPeakMB, unit: "MB")
        printMetric("Per Doc", baseline: baseline.memoryPerDocKB, comparison: comparison.memoryPerDocKB, unit: "KB")

        // Initialization Comparison
        print("\nInitialization Time (Comparison / Baseline):")
        printMetric("Init Time", baseline: baseline.initTimeMs, comparison: comparison.initTimeMs, unit: "ms")

        print("=".repeating(count: 70) + "\n")
    }

    /// Print a summary table for multiple test runs.
    ///
    /// - Parameter results: Array of labeled metrics
    public func printSummaryTable(_ results: [(label: String, metrics: PerformanceMetrics)]) {
        guard !results.isEmpty else { return }

        print("\n" + "=".repeating(count: 100))
        print("Performance Summary")
        print("=".repeating(count: 100))

        // Header
        let header = String(format: "%-30s %8s %8s %8s %8s %10s %10s",
                           "Configuration", "Avg (ms)", "P95 (ms)", "P99 (ms)", "Init (ms)", "Peak (MB)", "Per Doc (KB)")
        print(header)
        print("-".repeating(count: 100))

        // Rows
        for (label, metrics) in results {
            let truncatedLabel = String(label.prefix(30))
            let row = String(
                format: "%-30s %8.2f %8.2f %8.2f %8.2f %10.2f %10.2f",
                truncatedLabel,
                metrics.avgLatency,
                metrics.p95Latency,
                metrics.p99Latency,
                metrics.initTimeMs,
                metrics.memoryPeakMB,
                metrics.memoryPerDocKB
            )
            print(row)
        }

        print("=".repeating(count: 100) + "\n")
    }

    // MARK: - Private Helpers

    private func printMetric(_ name: String, baseline: Double, comparison: Double, unit: String) {
        let diff = comparison - baseline
        let percentChange = baseline > 0 ? (diff / baseline * 100) : 0
        let sign = diff >= 0 ? "+" : ""
        let indicator = percentChange > 5 ? "📈" : (percentChange < -5 ? "📉" : "➡️")

        print(String(
            format: "  %-12s %.2f %s / %.2f %s (%s%.1f%%) %@",
            name + ":",
            comparison, unit,
            baseline, unit,
            sign, percentChange,
            indicator
        ))
    }
}

// MARK: - String Extension

private extension String {
    /// Repeat a string N times.
    func repeating(count: Int) -> String {
        String(repeating: self, count: count)
    }
}

// MARK: - Int Extension

private extension Int {
    /// Format integer with thousands separators.
    func formatted() -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: self)) ?? "\(self)"
    }
}
