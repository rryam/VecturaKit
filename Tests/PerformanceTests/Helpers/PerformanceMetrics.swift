import Foundation

/// Container for performance measurement results.
public struct PerformanceMetrics: Sendable {
    /// Search latency measurements in nanoseconds.
    public let searchLatencies: [UInt64]

    /// Peak memory usage in bytes.
    public let memoryPeakBytes: UInt64

    /// Initialization time in nanoseconds.
    public let initializationTimeNanos: UInt64

    /// Document addition throughput (documents per second).
    public let addDocumentThroughput: Double?

    /// Search accuracy compared to baseline (0.0 to 1.0).
    /// Only applicable when comparing indexed mode to fullMemory baseline.
    public let accuracy: Double?

    /// Total number of documents in the dataset.
    public let documentCount: Int

    /// Memory strategy used for this test.
    public let strategyDescription: String

    public init(
        searchLatencies: [UInt64],
        memoryPeakBytes: UInt64,
        initializationTimeNanos: UInt64,
        addDocumentThroughput: Double? = nil,
        accuracy: Double? = nil,
        documentCount: Int,
        strategyDescription: String
    ) {
        self.searchLatencies = searchLatencies
        self.memoryPeakBytes = memoryPeakBytes
        self.initializationTimeNanos = initializationTimeNanos
        self.addDocumentThroughput = addDocumentThroughput
        self.accuracy = accuracy
        self.documentCount = documentCount
        self.strategyDescription = strategyDescription
    }

    // MARK: - Computed Statistics

    /// Median (P50) search latency in milliseconds.
    public var p50Latency: Double {
        percentile(0.5, of: searchLatencies)
    }

    /// P95 search latency in milliseconds.
    public var p95Latency: Double {
        percentile(0.95, of: searchLatencies)
    }

    /// P99 search latency in milliseconds.
    public var p99Latency: Double {
        percentile(0.99, of: searchLatencies)
    }

    /// Average search latency in milliseconds.
    public var avgLatency: Double {
        guard !searchLatencies.isEmpty else { return 0 }
        let sum = searchLatencies.reduce(0, +)
        return Double(sum) / Double(searchLatencies.count) / 1_000_000.0
    }

    /// Initialization time in milliseconds.
    public var initTimeMs: Double {
        Double(initializationTimeNanos) / 1_000_000.0
    }

    /// Peak memory usage in megabytes.
    public var memoryPeakMB: Double {
        Double(memoryPeakBytes) / 1_024.0 / 1_024.0
    }

    /// Average memory per document in kilobytes.
    public var memoryPerDocKB: Double {
        guard documentCount > 0 else { return 0 }
        return Double(memoryPeakBytes) / Double(documentCount) / 1_024.0
    }

    // MARK: - Private Helpers

    private func percentile(_ percentileValue: Double, of values: [UInt64]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = Int(Double(sorted.count) * percentileValue)
        let clampedIndex = min(index, sorted.count - 1)
        return Double(sorted[clampedIndex]) / 1_000_000.0  // Convert to milliseconds
    }
}

/// Utility for measuring performance metrics during tests.
public actor PerformanceMonitor {
    private var startTime: UInt64 = 0
    private var searchLatencies: [UInt64] = []
    private var baselineMemory: UInt64 = 0
    private var peakMemory: UInt64 = 0

    public init() {}

    /// Start timing an operation.
    public func startTimer() {
        startTime = DispatchTime.now().uptimeNanoseconds
    }

    /// Record the elapsed time since startTimer() was called.
    /// - Returns: Elapsed time in nanoseconds
    @discardableResult
    public func recordElapsed() -> UInt64 {
        let elapsed = DispatchTime.now().uptimeNanoseconds - startTime
        searchLatencies.append(elapsed)
        return elapsed
    }

    /// Measure and return elapsed time without recording it.
    public func getElapsed() -> UInt64 {
        return DispatchTime.now().uptimeNanoseconds - startTime
    }

    /// Record a search latency measurement.
    public func recordSearchLatency(_ nanos: UInt64) {
        searchLatencies.append(nanos)
    }

    /// Update memory usage tracking.
    public func updateMemoryUsage() {
        let currentMemory = getCurrentMemoryUsage()
        if baselineMemory == 0 {
            baselineMemory = currentMemory
        }
        peakMemory = max(peakMemory, currentMemory)
    }

    /// Get current memory usage in bytes.
    private func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Build final metrics report.
    public func buildMetrics(
        initTimeNanos: UInt64,
        documentCount: Int,
        strategyDescription: String,
        addDocumentThroughput: Double? = nil,
        accuracy: Double? = nil
    ) -> PerformanceMetrics {
        updateMemoryUsage()  // Final memory check

        return PerformanceMetrics(
            searchLatencies: searchLatencies,
            memoryPeakBytes: peakMemory - baselineMemory,
            initializationTimeNanos: initTimeNanos,
            addDocumentThroughput: addDocumentThroughput,
            accuracy: accuracy,
            documentCount: documentCount,
            strategyDescription: strategyDescription
        )
    }

    /// Reset all measurements.
    public func reset() {
        startTime = 0
        searchLatencies = []
        baselineMemory = 0
        peakMemory = 0
    }
}
