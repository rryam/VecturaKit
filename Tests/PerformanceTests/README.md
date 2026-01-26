# Performance Tests - Developer Guide

Complete guide for running, customizing, and understanding VecturaKit performance tests.

**⚠️ Important: Memory-Optimized Configuration**

These tests have been optimized to work on systems with limited memory (e.g., 8-16GB RAM). Due to CoreML memory constraints, dataset sizes have been reduced from the original 10K-5K range to 1K-3K range. This ensures tests can complete successfully on most development machines.

For systems with > 32GB RAM, you can increase dataset sizes by editing the test files directly.

---

## Quick Start

### Run Your First Test (6 seconds)

```bash
swift test --filter BenchmarkSuite.smallDatasetFullMemory
```

**Expected Output:**
```
======================================================================
Performance Report: 1K docs - fullMemory
======================================================================

Dataset:
  Documents: 1,000

Search Latency:
  Average: 10.43 ms  ⭐⭐⭐⭐⭐
  P50:     10.38 ms
  P95:     10.98 ms
  P99:     11.37 ms

Memory Usage:
  Peak:     182.80 MB
  Per Doc:  187.18 KB

Throughput:
  Add Documents: 178 docs/sec
======================================================================
```

**Success!** You've just benchmarked VecturaKit's fullMemory strategy.

---

## Structure

```
Tests/PerformanceTests/
├── README.md                      # This file
│
├── BenchmarkSuite.swift           # Core benchmarks (7 tests)
├── ScalabilitySuite.swift         # Scaling analysis (5 tests)
├── MemoryProfilerSuite.swift      # Memory profiling (5 tests)
├── ParameterTuningSuite.swift     # Parameter optimization (5 tests)
├── AccuracyTests.swift            # Search quality (4 tests)
│
├── Helpers/
│   ├── PerformanceMetrics.swift   # Metrics collection
│   ├── TestDataGenerator.swift    # Data generation
│   └── ResultsReporter.swift      # Report formatting
│
└── TestData/
    └── README.md                  # Custom data guide
```

**Total: 26 performance tests**

### Embedder Selection (Speed vs Realism)

By default, performance tests use a deterministic embedder to keep runtimes short and avoid
downloading CoreML models. To run with real embeddings, set:

```bash
VECTURA_PERF_USE_SWIFT_EMBEDDER=1
```

---

## Running Tests

### By Test Suite

```bash
# Core benchmarks (~2 min)
# Tests 1K-3K documents with fullMemory and automatic strategies
swift test --filter BenchmarkSuite

# Scalability analysis (~8 min)
# Tests scaling from 500-1.5K documents
swift test --filter ScalabilitySuite

# Memory profiling (~5 min)
# Analyzes memory usage with reduced dataset sizes
swift test --filter MemoryProfilerSuite

# Parameter tuning (~10 min)
# Tests with 1.5K documents to optimize parameters
swift test --filter ParameterTuningSuite

# Accuracy validation (~5 min)
# Validates search quality with 1K documents
swift test --filter AccuracyTests

# All performance tests (~30 min)
# Note: May fail on memory-constrained systems
swift test --filter PerformanceTests
```

**⚠️ Memory Constraints Warning:**
- Running all tests together may exceed available memory
- Consider running test suites individually
- Monitor Activity Monitor during execution
- Close other applications before running tests

### Individual Tests

```bash
# Small dataset benchmarks
swift test --filter BenchmarkSuite.smallDatasetFullMemory
swift test --filter BenchmarkSuite.smallDatasetAutomatic

# Strategy comparison
swift test --filter BenchmarkSuite.compareStrategies10K

# Parameter tuning
swift test --filter ParameterTuningSuite.candidateMultiplierSweep
swift test --filter ParameterTuningSuite.batchSizeSweep

# Memory analysis
swift test --filter MemoryProfilerSuite.fullMemoryLifecycle
swift test --filter MemoryProfilerSuite.strategyMemoryComparison

# Accuracy testing
swift test --filter AccuracyTests.basicAccuracyTest
```

---

## Saving Results

### Save Test Results (Archived)

```bash
# Save benchmark results
swift test --filter BenchmarkSuite \
  > ArchivedResults/$(date +%Y-%m-%d)_benchmark.txt 2>&1

# Save parameter tuning
swift test --filter ParameterTuningSuite.candidateMultiplierSweep \
  | tee ArchivedResults/$(date +%Y-%m-%d)_tuning.txt
```

### Quick Save to Desktop

```bash
# For quick local testing
swift test --filter BenchmarkSuite > ~/Desktop/benchmark_$(date +%Y%m%d_%H%M).txt 2>&1
```

---

## Test Suites Overview

### 1. BenchmarkSuite (7 tests)

**Purpose:** Establish performance baselines for different strategies and dataset sizes.

**Key Tests:**
- `smallDatasetFullMemory` - 1K docs, fullMemory baseline
- `smallDatasetAutomatic` - 1K docs, automatic strategy
- `mediumDatasetFullMemory` - 3K docs, fullMemory (optimized from 10K)
- `mediumDatasetIndexed` - 3K docs, indexed mode (optimized from 10K)
- `compareStrategies10K` - Direct comparison at 2K docs (optimized from 10K)
- `candidateMultiplierComparison` - Test mult: 5, 10, 15, 20 (at 2K docs)
- `batchSizeComparison` - Test batch: 50, 100, 150, 200 (at 2K docs)

**⚠️ Dataset Size Reduction:**
All tests have been optimized for memory-constrained systems. Original 10K document tests reduced to 2K-3K to avoid CoreML memory exhaustion.

**Typical Output:**
```
Performance Report: 1K docs - fullMemory
  Search Latency:
    Average: 10.43 ms
    P99:     11.37 ms
  Memory:
    Peak:    182.80 MB
    Per Doc: 187.18 KB
  Throughput: 178 docs/sec
```

**When to Run:**
- Before/after code changes
- Establishing baseline for new hardware
- Validating performance improvements

---

### 2. ScalabilitySuite (5 tests)

**Purpose:** Analyze how strategies scale from small to large datasets.

**Key Tests:**
- `fullMemoryScaling` - Growth curves for fullMemory (500, 1K, 1.5K)
- `indexedStrategyScaling` - Growth curves for indexed (500, 1K, 1.5K)
- `crossStrategyScaling` - Side-by-side comparison
- `addDocumentThroughputScaling` - Throughput analysis (500, 1K)
- `memoryGrowthPatterns` - Memory usage patterns (500, 1K, 1.5K)

**⚠️ Dataset Size Reduction:**
Reduced from [1K, 2K, 5K] to [500, 1K, 1.5K] to accommodate CoreML memory limits.

**Use Cases:**
- Understanding scaling characteristics
- Planning for dataset growth
- Identifying performance inflection points

**Sample Output:**
```
Performance Summary
Configuration                    Avg (ms) Memory (MB)
1000 docs                          10.43      182.80
2000 docs                          18.52      365.24
5000 docs                          42.18      912.45
```

---

### 3. MemoryProfilerSuite (5 tests)

**Purpose:** Deep memory usage analysis throughout lifecycle.

**Key Tests:**
- `fullMemoryLifecycle` - Memory tracking through init → add → search (1K docs, 30 searches)
- `indexedStrategyLifecycle` - Same for indexed mode (1K docs, 30 searches)
- `strategyMemoryComparison` - Direct memory efficiency comparison (1K docs)
- `searchMemoryImpact` - Memory delta during 50 searches (500 docs)
- `batchSizeMemoryImpact` - How batch size affects memory peaks (1K docs, 2 batch sizes)

**⚠️ Dataset Size Reduction:**
Reduced from 2K-3K documents to 500-1K documents to prevent memory exhaustion.

**Sample Output:**
```
Memory Lifecycle:
Phase                     Memory (MB)    Delta (MB)
Baseline                    245.32         0.00
After Init                  246.18         0.86
After 1K docs               428.50       183.18
After 3K docs               612.24       366.92
After 50 searches           615.82       370.50
```

**Use Cases:**
- Detecting memory leaks
- Optimizing memory footprint
- Understanding memory patterns

---

### 4. ParameterTuningSuite (5 tests)

**Purpose:** Find optimal configurations for indexed mode.

**Key Tests:**
- `candidateMultiplierSweep` - Test mult: 5, 10, 15, 20 (1.5K docs, 20 queries)
- `batchSizeSweep` - Test batch: 50, 100, 150 (1.5K docs, 20 queries)
- `maxConcurrentBatchesSweep` - Test concurrency: 2, 4, 6 (1.5K docs, 20 queries)
- `optimalParameterCombination` - Multi-dimensional optimization (3 configs, 1.5K docs)
- `workloadSpecificRecommendations` - Preset configurations (3 workloads, 1.5K docs)

**⚠️ Dataset Size Reduction:**
Reduced from 3K to 1.5K documents and reduced number of test configurations to save memory.

**Parameter Recommendations:**

| Workload | candidateMultiplier | batchSize | maxConcurrentBatches |
|----------|--------------------:|----------:|---------------------:|
| Low-latency | 5 | 50 | 6 |
| Memory-constrained | 10 | 50 | 2 |
| High-accuracy | 20 | 100 | 4 |
| High-throughput | 10 | 200 | 6 |
| **Balanced (default)** | **10** | **100** | **4** |

**Use Cases:**
- Tuning for specific workloads
- Optimizing accuracy vs speed trade-offs
- Finding memory-optimal configurations

---

### 5. AccuracyTests (4 tests)

**Purpose:** Validate search quality in indexed mode vs fullMemory.

**Key Tests:**
- `basicAccuracyTest` - Recall@10 measurement at 1K docs (15 queries)
- `candidateMultiplierAccuracy` - Accuracy across mult: 5, 10, 15 (1K docs, 10 queries)
- `accuracyPerformanceTradeoff` - Recall vs latency analysis (1K docs, 10 queries)
- `rankingQualityTest` - Position preservation rate (1K docs, 10 queries)

**⚠️ Dataset Size Reduction:**
Reduced from 2K to 1K documents and reduced query counts to prevent memory issues.

**Expected Results (with IndexedVecturaStorage):**

| Multiplier | Recall@10 | Avg Latency | Notes |
|-----------:|----------:|------------:|-------|
| 5 | 60-70% | ~30 ms | ⚠️ **estimated for production ANN** |
| 10 | 80-90% | ~40 ms | ⚠️ **estimated for production ANN** |
| 15 | 90-95% | ~50 ms | ⚠️ **estimated for production ANN** |
| 20 | 95-99% | ~60 ms | ⚠️ **estimated for production ANN** |

**⚠️ Important Notes:**
- Current tests use `MockIndexedStorage` which achieves **100% recall** by computing exact similarities on all documents
- This is not representative of production ANN algorithms (HNSW, IVF, PQ) which trade accuracy for speed
- Real ANN implementations will have lower recall (typically 85-98% at best)
- The estimates above are based on typical ANN algorithm behavior and require validation with a production `IndexedVecturaStorage` implementation
- See `Tests/PerformanceTests/Helpers/MockIndexedStorage.swift` for implementation details

**Use Cases:**
- Validating indexed mode accuracy
- Choosing optimal candidateMultiplier
- Understanding recall trade-offs

---

## Customizing Tests

### Modify Dataset Sizes

Edit test parameters directly:

```swift
// In BenchmarkSuite.swift
let metrics = try await runBenchmark(
    documentCount: 2_000,  // ← Change this
    queryCount: 100,       // ← Change this
    strategy: .fullMemory,
    strategyName: "2K docs - fullMemory"
)
```

### Add Custom Test

```swift
@Test("My custom benchmark")
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func myCustomBenchmark() async throws {
    let generator = TestDataGenerator()
    let monitor = PerformanceMonitor()

    // Generate test data
    let documents = generator.generateDocuments(count: 1_000, seed: 12345)

    // Setup VecturaKit
    let config = VecturaConfig(name: "test-db", memoryStrategy: .fullMemory)
    let vectura = try await VecturaKit(
        config: config,
        embedder: PerformanceTestConfig.makeEmbedder()
    )

    // Measure performance
    await monitor.startTimer()
    _ = try await vectura.addDocuments(texts: documents)
    await monitor.recordElapsed()

    // Build and print results
    let metrics = await monitor.buildMetrics(...)
    let reporter = ResultsReporter()
    reporter.printReport(metrics)
}
```

### Use Custom Test Data

See [TestData/README.md](./TestData/README.md) for using external datasets.

---

## Understanding Results

### Latency Metrics

| Metric | Meaning | Target (1K docs) |
|--------|---------|------------------|
| **Average** | Mean search time | < 20 ms |
| **P50** | 50% of searches faster | < 15 ms |
| **P95** | 95% of searches faster | < 25 ms |
| **P99** | 99% of searches faster | < 30 ms |

**Focus on P95/P99** for understanding worst-case user experience.

### Memory Metrics

| Metric | Meaning | Typical Value |
|--------|---------|---------------|
| **Peak** | Maximum RAM usage | ~180-200 MB (1K docs) |
| **Per Doc** | Memory per document | ~150-200 KB |
| **Delta** | Growth from baseline | Varies by operation |

### Performance Ratings

- ⭐⭐⭐⭐⭐ **Excellent:** < 10 ms latency
- ⭐⭐⭐⭐ **Good:** 10-20 ms
- ⭐⭐⭐ **Fair:** 20-50 ms
- ⭐⭐ **Poor:** 50-100 ms
- ⭐ **Needs optimization:** > 100 ms

---

## Troubleshooting

### Tests Crash (Signal 11)

**Symptom:** `error: Exited with unexpected signal code 11`

**Root Cause:** CoreML memory exhaustion when processing embeddings

**Common Scenarios:**
- Running multiple tests in sequence without cleanup
- Testing with datasets > 1.5K documents on 8-16GB RAM systems
- Multiple test suites running concurrently
- System memory pressure from other applications

**Solutions:**

1. **Run smaller dataset tests:**
   ```bash
   swift test --filter BenchmarkSuite.smallDatasetFullMemory
   swift test --filter BenchmarkSuite.smallDatasetAutomatic
   ```

2. **Run tests individually (not suites):**
   ```bash
   # Instead of running entire BenchmarkSuite
   swift test --filter "BenchmarkSuite.smallDatasetFullMemory"
   # Wait for completion, then:
   swift test --filter "BenchmarkSuite.smallDatasetAutomatic"
   ```

3. **Free up system memory:**
   - Close Safari, Chrome, and other memory-intensive apps
   - Check Activity Monitor for memory pressure
   - Ensure > 4GB free RAM before running tests
   - Consider restarting your Mac if memory is fragmented

4. **Reduce dataset sizes further:**
   Edit test files to use even smaller datasets (e.g., 500 documents)

5. **Monitor memory during execution:**
   ```bash
   # In Terminal 1:
   watch -n 1 'top -l 1 | grep -A 5 "PhysMem"'

   # In Terminal 2:
   swift test --filter BenchmarkSuite.smallDatasetFullMemory
   ```

6. **Use release builds (slightly more memory efficient):**
   ```bash
   swift test -c release --filter BenchmarkSuite.smallDatasetFullMemory
   ```

**If problems persist:**
- Your system may need > 16GB RAM for larger tests
- Consider skipping scalability and parameter tuning tests
- Focus on small dataset benchmarks (< 1K documents)

---

### Slow Test Execution

**Symptom:** Tests take much longer than expected

**Causes:**
- Debug build overhead
- System under heavy load
- Disk I/O bottleneck

**Solutions:**
```bash
# Use release build for accurate benchmarking
swift test -c release --filter BenchmarkSuite

# Close background apps
# Use SSD for database directory
# Run during low system load
```

---

### Accuracy Tests Show 100% Recall

**Symptom:** `AccuracyTests` report 100% recall

**Explanation:** This is expected behavior with `MockIndexedStorage`

**Why 100% recall?**
- `MockIndexedStorage` computes exact cosine similarity for all documents
- It then returns the top-N most similar documents
- This is perfect accuracy but not representative of real ANN algorithms

**What about production systems?**
Real ANN implementations (HNSW, IVF, PQ) will have lower recall:
- They use approximations to achieve fast search on large datasets
- Typical recall ranges: 85-98% (depending on algorithm and tuning)
- Higher `candidateMultiplier` improves recall at the cost of latency

**For testing production accuracy:**
Implement a real `IndexedVecturaStorage` with an ANN library like:
- FAISS (Facebook AI Similarity Search)
- Annoy (Approximate Nearest Neighbors Oh Yeah)
- hnswlib (HNSW implementation)

See `Tests/PerformanceTests/Helpers/MockIndexedStorage.swift` for the current mock implementation.

```swift
actor MyIndexedStorage: IndexedVecturaStorage {
    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument] {
        // Your implementation
    }

    func getTotalDocumentCount() async throws -> Int {
        // Your implementation
    }

    func searchCandidates(queryEmbedding: [Float], topK: Int, prefilterSize: Int) async throws -> [UUID] {
        // Your implementation
    }

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
        // Your implementation
    }
}
```

---

### No Output Displayed

**Symptom:** Test passes but no performance report shown

**Cause:** Output captured by test framework

**Solution:**
```bash
# Redirect stderr to stdout
swift test --filter BenchmarkSuite 2>&1

# Or save to file
swift test --filter BenchmarkSuite > results.txt 2>&1
```

---

## Common Workflows

### Daily Development

```bash
# Quick sanity check (~6 sec)
swift test --filter BenchmarkSuite.smallDatasetFullMemory
```

### Before Pull Request

```bash
# Run core benchmarks (~5 min)
swift test --filter BenchmarkSuite

# Save results for comparison
swift test --filter BenchmarkSuite \
  > ArchivedResults/pr_$(date +%Y%m%d).txt 2>&1
```

### Performance Investigation

```bash
# Memory analysis
swift test --filter MemoryProfilerSuite.fullMemoryLifecycle

# Scaling issues
swift test --filter ScalabilitySuite.fullMemoryScaling

# Parameter optimization
swift test --filter ParameterTuningSuite.candidateMultiplierSweep
```

### Production Tuning

```bash
# Find optimal parameters
swift test --filter ParameterTuningSuite.optimalParameterCombination

# Validate accuracy trade-offs
swift test --filter AccuracyTests.accuracyPerformanceTradeoff
```

### Release Validation

```bash
# Full suite (~1 hour)
swift test --filter PerformanceTests

# Save comprehensive results
swift test --filter PerformanceTests \
  > ArchivedResults/release_$(date +%Y%m%d).txt 2>&1
```

---

## Pre-Commit Checklist

Before committing performance-related changes:

- [ ] Run core benchmarks: `swift test --filter BenchmarkSuite`
- [ ] No performance regressions > 10% (latency increase or memory increase)
- [ ] Save baseline results if performance improved significantly
- [ ] Update `Docs/TEST_RESULTS_SUMMARY.md` if recommendations change
- [ ] Document parameter changes in commit message

---

## Helper Classes

### PerformanceMetrics

Collects and computes performance statistics:

```swift
let monitor = PerformanceMonitor()

await monitor.startTimer()
// ... operation ...
await monitor.recordElapsed()

await monitor.updateMemoryUsage()

let metrics = await monitor.buildMetrics(
    initTimeNanos: initTime,
    documentCount: 1000,
    strategyDescription: "1K docs - fullMemory"
)
```

**Computed Statistics:**
- `p50Latency`, `p95Latency`, `p99Latency` - Percentile latencies
- `avgLatency` - Average latency
- `memoryPeakMB` - Peak memory in MB
- `memoryPerDocKB` - Memory per document in KB

---

### TestDataGenerator

Generates reproducible test data:

```swift
let generator = TestDataGenerator()

// Generate documents (reproducible with seed)
let documents = generator.generateDocuments(count: 1_000, seed: 12345)

// Generate queries
let queries = generator.generateQueries(count: 50, seed: 54321)

// Generate large dataset with custom word count
let largeDataset = generator.generateLargeDataset(
    documentCount: 10_000,
    avgWordsPerDoc: 50,
    seed: 98765
)
```

**Features:**
- Reproducible with seeds
- Diverse topics (20 topics × 10 contexts × 10 domains)
- Configurable document characteristics

---

### ResultsReporter

Formats and prints results:

```swift
let reporter = ResultsReporter()

// Single test report
reporter.printReport(metrics)

// Comparison report
reporter.printComparison(baselineMetrics, vs: candidateMetrics)

// Summary table for multiple configs
reporter.printSummaryTable([
    (label: "Config A", metrics: metricsA),
    (label: "Config B", metrics: metricsB)
])
```

---

## Related Documentation

### For Users
- **[Documentation Hub](../../Docs/)** - All VecturaKit documentation
- **[Performance Results](../../Docs/TEST_RESULTS_SUMMARY.md)** - Detailed test results for users
- **[Indexed Storage Guide](../../Docs/INDEXED_STORAGE_GUIDE.md)** - Memory strategy implementation

### For Developers
- **[Test Data Guide](./TestData/README.md)** - Using custom test datasets
- **[Main README](../../README.md)** - Project overview

---

## Tips & Best Practices

### 1. Start Small
Run `smallDatasetFullMemory` first to verify your setup:
```bash
swift test --filter BenchmarkSuite.smallDatasetFullMemory
```

### 2. Monitor System Resources
- Use Activity Monitor during large tests
- Close unnecessary applications
- Ensure sufficient free RAM (4GB+ recommended)

### 3. Save Results for Comparison
Always save baseline results before making changes:
```bash
swift test --filter BenchmarkSuite > baseline_$(date +%Y%m%d).txt 2>&1
```

### 4. Run Multiple Times
For consistent results, run critical tests 3-5 times:
```bash
for i in {1..3}; do
  swift test --filter BenchmarkSuite.smallDatasetFullMemory
done
```

### 5. Document System Specs
When sharing results, note:
- Hardware (Apple Silicon model, RAM)
- OS version
- Other running processes
- Build configuration (debug/release)

---

## FAQ

**Q: How long do tests take?**
- Small tests (1K docs): 5-15 seconds
- Medium tests (2-3K docs): 30-60 seconds (may fail on low-memory systems)
- Scalability tests: 5-10 minutes (reduced dataset sizes)
- Full suite: 20-30 minutes (may fail due to cumulative memory usage)

**Q: Why were dataset sizes reduced?**
A: CoreML has significant memory overhead when processing embeddings. On systems with 8-16GB RAM, tests with > 1.5K documents often crash with Signal 11. The reduced sizes ensure tests can complete successfully on most development machines.

**Q: Can I increase dataset sizes for more realistic testing?**
A: Yes, if you have > 32GB RAM:
1. Edit test files directly (e.g., `BenchmarkSuite.swift`)
2. Change `documentCount` parameters (e.g., from 1_000 to 10_000)
3. Increase `queryCount` proportionally
4. Monitor memory usage carefully

**Q: Can I run tests in parallel?**
A: Not recommended for performance tests due to memory constraints. Swift Testing runs tests concurrently by default, but this can cause memory exhaustion. Consider using `.serialized` trait or running tests individually.

**Q: Why do some tests fail?**
A: Common reasons:
- **Signal 11:** CoreML memory exhaustion (see Troubleshooting)
- **0% recall:** `FileStorageProvider` doesn't implement `IndexedVecturaStorage` (expected)
- **Timeout:** System under heavy load or insufficient resources

**Q: How do I add custom test data?**
A: See [TestData/README.md](./TestData/README.md) for instructions.

**Q: Can I use these tests for CI?**
A: Yes! Add to your CI workflow:
```yaml
- name: Run Performance Tests
  run: swift test --filter BenchmarkSuite
```

**Q: How accurate are the benchmarks?**
A: Very accurate when run in release mode on a quiet system. Debug builds add ~2-3x overhead.

---

## Support

### Questions?

- **Test execution issues:** Check Troubleshooting section above
- **Custom test development:** Review Helper classes and examples
- **Performance results interpretation:** See "Understanding Results" section
- **User-facing performance docs:** See [Docs/TEST_RESULTS_SUMMARY.md](../../Docs/TEST_RESULTS_SUMMARY.md)

---

**Framework Version:** 1.0
**Total Tests:** 26
**Dataset Sizes:** Optimized for 8-16GB RAM systems (1K-3K documents)
**Last Updated:** 2025-10-30

**Memory Requirements:**
- Minimum: 8GB RAM (small tests only)
- Recommended: 16GB RAM (most tests)
- Ideal: 32GB+ RAM (full suite with increased dataset sizes)

**Happy Testing! 🚀**
