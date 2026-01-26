# Test Data

This directory contains test datasets for performance testing.

## Current Test Data Strategy

VecturaKit performance tests use **programmatic data generation** by default via `TestDataGenerator`:

```swift
let generator = TestDataGenerator()
let documents = generator.generateDocuments(count: 1000, seed: 12345)
let queries = generator.generateQueries(count: 50, seed: 54321)
```

**Advantages:**
- [x] Reproducible (seeded random generation)
- [x] No external dependencies
- [x] Scalable to any size
- [x] Version controlled

## Using Custom Test Data

### 1. Prepare Your Dataset

Create a text file with one document per line:

```
Tests/PerformanceTests/TestData/custom_dataset.txt
```

Example content:
```
Machine learning is a subset of artificial intelligence
Natural language processing enables computers to understand text
Vector databases provide semantic search capabilities
...
```

### 2. Load in Tests

```swift
@Test("Custom dataset performance")
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func customDatasetTest() async throws {
    let dataPath = URL(fileURLWithPath: "Tests/PerformanceTests/TestData/custom_dataset.txt")
    let content = try String(contentsOf: dataPath)
    let documents = content.components(separatedBy: "\n").filter { !$0.isEmpty }

    // Use documents for testing
    let vectura = try await VecturaKit(config: config, embedder: embedder)
    _ = try await vectura.addDocuments(texts: documents)

    // Run benchmarks...
}
```

### 3. Queries File (Optional)

Create a separate file for test queries:

```
Tests/PerformanceTests/TestData/custom_queries.txt
```

Example:
```
machine learning
natural language
vector search
```

## Recommended Dataset Formats

### Small Dataset (1K - 5K docs)
- File: `small_dataset.txt`
- Use: Quick benchmarks, development testing
- Size: ~100-500 KB

### Medium Dataset (10K - 50K docs)
- File: `medium_dataset.txt`
- Use: Scalability testing
- Size: ~1-5 MB

### Large Dataset (100K+ docs)
- File: Large files should be downloaded separately
- Use: Stress testing, production validation
- Note: Don't commit large files to git

## .gitignore

Large test data files are gitignored to keep repository size manageable:

```gitignore
# Test data files
Tests/PerformanceTests/TestData/*.txt
Tests/PerformanceTests/TestData/*.json
Tests/PerformanceTests/TestData/*.csv

# Except README
!Tests/PerformanceTests/TestData/README.md
```

## Public Datasets

For realistic testing, consider these public datasets:

### 1. Wikipedia Abstracts
- Source: Wikipedia dumps
- Size: ~6M documents
- Download: https://dumps.wikimedia.org/

### 2. ArXiv Papers
- Source: ArXiv dataset
- Size: ~2M documents
- Download: https://www.kaggle.com/datasets/Cornell-University/arxiv

### 3. News Articles
- Source: Various news datasets
- Size: Varies
- Use: Realistic document diversity

## Best Practices

1. **Small files in git:** < 100 KB can be committed
2. **Large files external:** Download separately, document in README
3. **Reproducible seeds:** Use consistent random seeds for generated data
4. **Document characteristics:** Note avg length, vocabulary size, etc.
5. **Multiple sizes:** Maintain small/medium/large variants

## Data Generation Tips

### Diverse Documents

```swift
let generator = TestDataGenerator()

// Generate with different characteristics
let short = generator.generateDocuments(count: 1000, avgWordsPerDoc: 10, seed: 12345)
let medium = generator.generateDocuments(count: 1000, avgWordsPerDoc: 50, seed: 12345)
let long = generator.generateDocuments(count: 1000, avgWordsPerDoc: 200, seed: 12345)
```

### Realistic Queries

```swift
// Generate queries matching document topics
let queries = generator.generateQueries(count: 100, seed: 54321)

// Or use custom queries
let customQueries = [
    "machine learning algorithms",
    "neural network architecture",
    "deep learning frameworks"
]
```

## Example: Using Custom Data

```swift
@Test("Wikipedia abstracts benchmark")
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func wikipediaAbstractsTest() async throws {
    // Load Wikipedia abstracts (download separately)
    let dataPath = URL(fileURLWithPath: "Tests/PerformanceTests/TestData/wikipedia_1k.txt")

    guard FileManager.default.fileExists(atPath: dataPath.path) else {
        // Skip test if data not available
        print("Wikipedia test data not found, skipping...")
        return
    }

    let content = try String(contentsOf: dataPath)
    let documents = content.components(separatedBy: "\n").filter { !$0.isEmpty }

    let config = VecturaConfig(name: "wiki-test", memoryStrategy: .fullMemory)
    let vectura = try await VecturaKit(
        config: config,
        embedder: PerformanceTestConfig.makeEmbedder()
    )

    // Benchmark document addition
    let addStart = DispatchTime.now().uptimeNanoseconds
    _ = try await vectura.addDocuments(texts: documents)
    let addElapsed = DispatchTime.now().uptimeNanoseconds - addStart

    print("Added \(documents.count) Wikipedia abstracts in \(Double(addElapsed) / 1_000_000.0) ms")

    // Run searches...
}
```

## Getting Started

### Option 1: Use Generated Data (Recommended)

No setup required! Tests use `TestDataGenerator` by default.

```bash
swift test --filter BenchmarkSuite
```

### Option 2: Use Custom Data

1. Place your data files in `Tests/PerformanceTests/TestData/`
2. Create custom tests that load from files
3. Run your tests

```bash
swift test --filter MyCustomBenchmark
```

---

**Note:** Default tests use generated data and work out-of-the-box. Custom data is optional for specialized testing scenarios.
