# Indexed Storage Guide

> This feature is fully functional and backward-compatible, but performance
> characteristics are still being measured and optimized. We welcome feedback
> and real-world performance data from the community.

## Performance Data Reliability

Before using the performance metrics in this guide, understand their reliability:

| Data Category | Reliability | Verification Status |
|---------------|-------------|---------------------|
| **1K docs, fullMemory** | ✅ **High** | Measured on Apple M-series (debug build) |
| **10K+ docs, fullMemory** | ⚠️ **Medium** | Linear extrapolation (not yet measured) |
| **indexed mode recall** | ❓ **Unknown** | Based on typical ANN algorithms (HNSW, IVF) |
| **indexed mode performance** | ❓ **Unknown** | Requires `IndexedVecturaStorage` implementation |

**Key points:**
- Small dataset (1K docs) performance is well-tested and reliable
- Larger dataset estimates are based on linear scaling assumptions
- Indexed mode metrics assume a production ANN implementation (not yet available)
- `MockIndexedStorage` used in tests provides ideal (100% recall) but unrealistic performance

This guide explains how to use VecturaKit's indexed storage capabilities for handling large-scale datasets efficiently.

## Overview

VecturaKit now supports two memory management strategies:

1. **Full Memory Mode** (default): All documents loaded into RAM for maximum search speed
2. **Indexed Mode**: On-demand loading with storage-layer indexing for memory efficiency

## Memory Strategies

### Automatic Mode (Recommended)

The automatic mode selects the best strategy based on your dataset size:

```swift
// Automatic strategy (default)
let config = VecturaConfig(name: "my-database")
// config.memoryStrategy defaults to .automatic()

let vectura = try await VecturaKit(config: config)
```

**Behavior:**
- < 10,000 documents → Uses `fullMemory` mode
- ≥ 10,000 documents → Uses `indexed` mode (if storage supports it)

### Full Memory Mode

Explicitly use full memory mode for guaranteed fast searches:

```swift
let config = VecturaConfig(
    name: "my-database",
    memoryStrategy: .fullMemory
)

let vectura = try await VecturaKit(config: config)
```

**Best for:**
- Small to medium datasets (< 100,000 documents)
- Sub-10ms search latency requirements
- When memory usage is not a constraint

**Memory usage:** ~180-200 KB per document (with 384-dimensional embeddings)

### Indexed Mode

Use indexed mode for large datasets:

```swift
let config = VecturaConfig(
    name: "my-database",
    memoryStrategy: .indexed(
        candidateMultiplier: 10      // Search 10× topK candidates
    )
)

let vectura = try await VecturaKit(config: config)
```

**Best for:**
- Large datasets (> 100,000 documents)
- Memory-constrained environments
- When moderate search latency is acceptable

**Parameters:**
- `candidateMultiplier`: How many candidates to fetch before exact ranking (higher = better accuracy, slower)
- `batchSize`: Number of documents to load per batch during concurrent loading (default: 100)
- `maxConcurrentBatches`: Maximum number of concurrent batch loading operations (default: 4)

**Memory usage (peak during search):**

During search operations, indexed mode loads candidate documents into memory for exact similarity computation:

```
Peak memory = topK × candidateMultiplier × avg_document_size
```

**Examples:**
- `topK=10, candidateMultiplier=10`: ~0.4-0.5 MB (100 documents in memory)
- `topK=100, candidateMultiplier=10`: ~4-5 MB (1,000 documents in memory)
- `topK=100, candidateMultiplier=20`: ~8-10 MB (2,000 documents in memory)

**Note:** The actual memory footprint depends on:
- Document text length
- Embedding dimensions (default: 384 floats = 1.5 KB per document)
- Metadata size
- Storage provider's internal buffering

Between searches, memory is freed, so the baseline memory usage remains low.

## Storage Providers

### FileStorageProvider (Default)

The default `FileStorageProvider` only implements `VecturaStorage` and does not support indexed operations. When using `indexed` mode with `FileStorageProvider`, VecturaKit will automatically fall back to `fullMemory` mode.

```swift
// Uses FileStorageProvider by default
let vectura = try await VecturaKit(config: config)
```

### Custom Indexed Storage

To benefit from indexed mode, implement `IndexedVecturaStorage`:

```swift
public protocol IndexedVecturaStorage: VecturaStorage {
    // Pagination
    func loadDocuments(offset: Int, limit: Int) async throws -> [VecturaDocument]
    func getTotalDocumentCount() async throws -> Int

    // Vector indexing
    func searchCandidates(
        queryEmbedding: [Float],
        topK: Int,
        prefilterSize: Int
    ) async throws -> [UUID]

    func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument]
}
```

## Example: SQLite Storage Provider (Conceptual)

Here's a conceptual example of implementing indexed storage with SQLite:

```swift
import SQLite3

public final class SQLiteIndexedStorageProvider: IndexedVecturaStorage {
    private var db: OpaquePointer?

    public init(dbPath: String) throws {
        // Open database
        sqlite3_open(dbPath, &db)
        try createTables()
    }

    // MARK: - VecturaStorage

    public func loadDocuments() async throws -> [VecturaDocument] {
        // Full load (discouraged for large datasets)
        return try await loadDocuments(offset: 0, limit: Int.max)
    }

    public func saveDocument(_ document: VecturaDocument) async throws {
        // INSERT INTO documents (id, text, embedding) VALUES (?, ?, ?)
        // Update vector index
    }

    // MARK: - IndexedVecturaStorage

    public func getTotalDocumentCount() async throws -> Int {
        // SELECT COUNT(*) FROM documents
        return 0 // placeholder
    }

    public func searchCandidates(
        queryEmbedding: [Float],
        topK: Int,
        prefilterSize: Int
    ) async throws -> [UUID] {
        // Option 1: Use sqlite-vss extension for vector search
        // Option 2: Implement IVF (Inverted File) indexing
        // Option 3: Use Product Quantization (PQ)

        // Returns candidate document IDs
        return []
    }

    public func loadDocuments(ids: [UUID]) async throws -> [UUID: VecturaDocument] {
        // SELECT * FROM documents WHERE id IN (?, ?, ...)
        return [:]
    }
}
```

## Performance Comparison

> 📊 **For detailed benchmark results and methodology**, see [Performance Test Results](./TEST_RESULTS_SUMMARY.md).

The following table shows measured and estimated performance characteristics:

| Dataset Size | Strategy | Memory Usage | Init Time | Search Latency (Avg) | Data Source |
|--------------|----------|--------------|-----------|---------------------|-------------|
| 1K docs | fullMemory | 180-183 MB | 0.5-1.1 ms | 10-11 ms | ✅ **Measured** (Apple M-series, 384-dim) |
| 10K docs | fullMemory | ~1.8 GB | ~5-10 ms | ~100 ms | ⚠️ **Extrapolated** (linear scaling, not verified) |
| 100K docs | fullMemory | ~18 GB | ~50-100 ms | ~1000 ms | ⚠️ **Extrapolated** (may hit system limits) |
| 100K docs | indexed | Variable* | Variable | Variable | ❓ **Implementation-dependent** |
| 1M docs | indexed | Variable* | Variable | Variable | ❓ **Implementation-dependent** |

**Data reliability:**
- ✅ **Measured**: Actual benchmark results (see [TEST_RESULTS_SUMMARY.md](./TEST_RESULTS_SUMMARY.md))
  - Test environment: Debug build, macOS on Apple Silicon, 384-dimensional embeddings
  - Performance may be 20-30% better in release builds
- ⚠️ **Extrapolated**: Linear scaling assumptions from measured 1K baseline
  - Not yet verified with actual tests due to memory constraints
  - Actual values may differ based on hardware, caching, and system load
- ❓ **Implementation-dependent**: Requires `IndexedVecturaStorage` implementation
  - Current `FileStorageProvider` falls back to `fullMemory` mode
  - Performance depends on chosen ANN algorithm (HNSW, IVF, PQ, etc.)
  - Storage layer characteristics (SQLite, PostgreSQL, etc.)

*Memory usage for indexed mode depends on:
- Candidate pool size (`candidateMultiplier × topK`)
- Storage provider's internal buffering
- Vector index overhead (if using HNSW, IVF, etc.)

**Recommendation:** Profile your specific use case before making optimization decisions. See [TEST_RESULTS_SUMMARY.md](./TEST_RESULTS_SUMMARY.md) for detailed analysis.

## Migration Guide

### Existing Code (Still Works)

```swift
// No changes needed - automatic backward compatibility
let config = VecturaConfig(name: "my-db")
let vectura = try await VecturaKit(config: config)

// Searches work exactly as before
let results = try await vectura.search(
    query: "machine learning",
    numResults: 10
)
```

### Opt-in to Indexed Mode

```swift
// 1. Configure indexed strategy
let config = VecturaConfig(
    name: "my-db",
    memoryStrategy: .indexed()
)

// 2. Provide custom storage (when available)
let sqliteProvider = try SQLiteIndexedStorageProvider(dbPath: "/path/to/db")
let vectura = try await VecturaKit(config: config, storageProvider: sqliteProvider)

// 3. Use as normal - API unchanged
let results = try await vectura.search(
    query: "machine learning",
    numResults: 10
)
```

## Implementation Status

### ✅ Completed
- `IndexedVecturaStorage` protocol definition
- `VecturaConfig.MemoryStrategy` enum
- VecturaKit initialization refactoring
- Indexed search logic with automatic fallback
- Backward compatibility with existing code

### 🚧 Future Work
- Reference SQLiteIndexedStorageProvider implementation
- Vector indexing algorithms (IVF, HNSW, PQ)
- Document-level LRU caching for indexed mode
- Performance benchmarks and real-world testing
- GPU-accelerated search (MLX integration)

## FAQ

### Q: Does indexed mode work with FileStorageProvider?

**A:** No, `FileStorageProvider` only implements `VecturaStorage`. If you configure `indexed` mode with `FileStorageProvider`, VecturaKit will automatically fall back to `fullMemory` mode.

### Q: How accurate is indexed search?

**A:** Accuracy depends on `candidateMultiplier` and the underlying ANN (Approximate Nearest Neighbor) algorithm used by your `IndexedVecturaStorage` implementation.

**Theoretical estimates** (based on typical ANN algorithms like HNSW/IVF):
- `candidateMultiplier: 5` → ~90% recall (⚠️ **estimated, not measured**)
- `candidateMultiplier: 10` → ~95% recall (⚠️ **estimated, not measured**)
- `candidateMultiplier: 20` → ~98% recall (⚠️ **estimated, not measured**)

**Important notes:**
- These estimates assume a production-grade ANN index (HNSW, IVF, etc.)
- `MockIndexedStorage` (used in tests) achieves 100% recall because it performs exact similarity computation on all documents, which is not representative of real ANN performance
- Actual recall will vary based on:
  - Dataset characteristics (size, distribution, dimensionality)
  - ANN algorithm choice and tuning
  - Index build parameters

The second-stage exact ranking ensures results within the candidate pool are perfectly sorted, but the quality of candidates depends on the ANN algorithm's effectiveness.

### Q: Can I switch strategies after initialization?

**A:** No, the strategy is fixed at initialization. To switch, create a new `VecturaKit` instance with the desired configuration.

### Q: Does this break existing code?

**A:** No! All existing code continues to work without modifications. The default behavior is identical to the previous version.

## Contributing

To implement a custom indexed storage provider:

1. Conform to `IndexedVecturaStorage` protocol
2. Implement efficient pagination (`loadDocuments(offset:limit:)`)
3. Implement vector indexing (`searchCandidates(...)`)
4. Consider adding to VecturaKit as an official provider

See `Sources/VecturaKit/IndexedVecturaStorage.swift` for detailed protocol documentation.
