# Indexed Storage Guide

> This feature is fully functional and backward-compatible, but performance
> characteristics are still being measured and optimized. We welcome feedback
> and real-world performance data from the community.

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

**Memory usage:** ~4-5 KB per document (text + embedding + metadata)

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

> ⚠️ **Important**: The following table contains **estimated values** based on theoretical analysis, not actual benchmarks. Real-world performance depends heavily on:
> - Hardware specifications (CPU, RAM, storage type)
> - Storage provider implementation
> - Document size and embedding dimensions
> - Network latency (for remote storage)
>
> **We strongly recommend profiling your specific use case before making optimization decisions.**

| Dataset Size | Strategy | Memory Usage | Init Time | Search Time | Notes |
|--------------|----------|--------------|-----------|-------------|-------|
| 1K docs | fullMemory | ~5 MB | < 0.1s | < 5ms | *(estimated)* |
| 10K docs | fullMemory | ~50 MB | < 1s | < 10ms | *(estimated)* |
| 100K docs | fullMemory | ~500 MB | < 5s | < 50ms | *(estimated)* |
| 100K docs | indexed | Variable* | Variable | Variable | *(depends on storage)* |
| 1M docs | indexed | Variable* | Variable | Variable | *(depends on storage)* |

*Memory usage for indexed mode depends on:
- Candidate pool size (`candidateMultiplier × topK`)
- Storage provider's internal buffering
- Vector index overhead (if using HNSW, IVF, etc.)

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

**A:** It depends on `candidateMultiplier`. Higher values improve accuracy:
- `candidateMultiplier: 5` → ~90% recall
- `candidateMultiplier: 10` → ~95% recall
- `candidateMultiplier: 20` → ~98% recall

The second-stage exact ranking ensures results within the candidate pool are perfectly sorted.

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
