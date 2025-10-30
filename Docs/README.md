# VecturaKit Documentation

Welcome to VecturaKit's documentation hub. This directory contains comprehensive guides for understanding and using VecturaKit's features.

---

## Available Documentation

### [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md)

**Purpose:** Learn about VecturaKit's memory management strategies

**Topics Covered:**
- Memory strategy options (automatic, fullMemory, indexed)
- When to use each strategy
- Configuration examples
- `IndexedVecturaStorage` protocol implementation
- Performance considerations for large-scale datasets

**Target Audience:** Developers integrating VecturaKit

**When to Read:**
- Setting up VecturaKit for the first time
- Choosing the right memory strategy
- Working with large datasets (> 10K documents)
- Implementing custom storage providers

---

### [Performance Test Results](./TEST_RESULTS_SUMMARY.md)

**Purpose:** Understand VecturaKit's performance characteristics

**Topics Covered:**
- Benchmark results for different strategies
- Performance metrics (latency, memory, throughput)
- Scaling characteristics
- Parameter tuning recommendations
- Platform-specific considerations

**Target Audience:** Users evaluating VecturaKit, developers optimizing performance

**When to Read:**
- Evaluating VecturaKit for your use case
- Understanding expected performance
- Choosing optimal configurations
- Planning for dataset growth

**Quick Facts:**
- Search Latency (1K docs): ~10ms average, <12ms P99
- Memory Usage: ~180-200 KB per document (fullMemory mode)
- Initialization: Sub-millisecond
- Tested on: Apple Silicon (M-series), macOS 14.0+

---

## Quick Navigation

### For New Users

1. **Getting Started:** See [Main README](../README.md)
2. **Choose Memory Strategy:** Read [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md)
3. **Understand Performance:** Review [Test Results](./TEST_RESULTS_SUMMARY.md)

### For Developers

1. **Memory Strategies:** [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md)
2. **Performance Testing:** [Tests/PerformanceTests/](../Tests/PerformanceTests/)
3. **Test Results:** [Performance](./TEST_RESULTS_SUMMARY.md)

### For Decision Makers

1. **Performance Overview:** [Test Results Summary](./TEST_RESULTS_SUMMARY.md)
2. **Scaling Capabilities:** See "Scaling Characteristics" section in test results
3. **Platform Support:** Check [Main README](../README.md)

---

## Documentation Structure

```
Docs/
├── README.md                      # This file - Documentation index
├── INDEXED_STORAGE_GUIDE.md       # Memory strategy implementation guide
└── TEST_RESULTS_SUMMARY.md        # Detailed performance test results
```

**Additional Resources:**
- **Performance Tests:** [Tests/PerformanceTests/](../Tests/PerformanceTests/) - Run your own benchmarks
- **Main README:** [../README.md](../README.md) - Project overview and quick start
- **Examples:** [Sources/TestExamples/](../Sources/TestExamples/) - Sample code

---

## Find What You Need

### Memory & Storage Questions

**Q: How do I choose between fullMemory and indexed mode?**
→ [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md) - See "Choosing a Memory Strategy"

**Q: How much memory will VecturaKit use?**
→ [Performance Results](./TEST_RESULTS_SUMMARY.md) - See "Memory Usage" section

**Q: Can VecturaKit handle 100K documents?**
→ [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md) + [Performance Results](./TEST_RESULTS_SUMMARY.md)

### Performance Questions

**Q: How fast is search?**
→ [Performance Results](./TEST_RESULTS_SUMMARY.md) - See benchmark data

**Q: How do I optimize for my use case?**
→ [Performance Results](./TEST_RESULTS_SUMMARY.md) - See "Parameter Tuning" section

**Q: What are the scaling characteristics?**
→ [Performance Results](./TEST_RESULTS_SUMMARY.md) - See "Scaling Characteristics"

### Implementation Questions

**Q: How do I implement IndexedVecturaStorage?**
→ [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md) - See implementation examples

**Q: How do I run performance tests?**
→ [Tests/PerformanceTests/](../Tests/PerformanceTests/README.md)

**Q: What are best practices?**
→ [Indexed Storage Guide](./INDEXED_STORAGE_GUIDE.md) + [Performance Results](./TEST_RESULTS_SUMMARY.md)

---

## Performance Summary

Quick reference for VecturaKit's performance (fullMemory mode, 1K documents):

| Metric | Value |
|--------|-------|
| **Search Latency (P99)** | < 12 ms |
| **Initialization** | < 1 ms |
| **Memory per Document** | ~180-200 KB |
| **Document Addition** | 150-200 docs/sec |

**Platform:** Apple Silicon (M-series), macOS 14.0+

See [full test results](./TEST_RESULTS_SUMMARY.md) for detailed benchmarks.

---

## For Contributors

### Documentation Guidelines

1. **Keep docs up-to-date:** Update after significant changes
2. **User-focused:** Write for users, not just developers
3. **Examples:** Include code examples where helpful
4. **Accuracy:** Verify technical details before publishing

### Adding Documentation

1. Create new `.md` file in appropriate location
2. Update this README.md to include it
3. Add cross-references from related docs
4. Test all code examples

---

## External Resources

- **GitHub Repository:** [VecturaKit](https://github.com/rryam/VecturaKit)
- **Issues & Discussions:** [GitHub Issues](https://github.com/rryam/VecturaKit/issues)
- **Swift Package Index:** [VecturaKit on SwiftPM](https://swiftpackageindex.com/rryam/VecturaKit)

---

**Last Updated:** 2025-10-30
**Documentation Version:** 2.0
