# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps through local vector storage and retrieval.

Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** uses `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings) for generating and managing embeddings. It features **Model2Vec** support with the 32M parameter model as default for fast static embeddings.

The framework offers `VecturaKit` as the core vector database with pluggable embedding providers. Use `SwiftEmbedder` for `swift-embeddings` integration, `MLXEmbedder` for Apple's MLX framework acceleration, or `NLContextualEmbedder` for Apple's NaturalLanguage framework with zero external dependencies.

It also includes CLI tools (`vectura-cli` and `vectura-mlx-cli`) for easily trying out the package.

<p align="center">
  <img src="https://img.shields.io/badge/Swift-6.0+-fa7343?style=flat&logo=swift&logoColor=white" alt="Swift 6.0+">
  <br>
  <img src="https://img.shields.io/badge/iOS-17.0+-000000?style=flat&logo=apple&logoColor=white" alt="iOS 17.0+">
  <img src="https://img.shields.io/badge/macOS-14.0+-000000?style=flat&logo=apple&logoColor=white" alt="macOS 14.0+">
  <img src="https://img.shields.io/badge/watchOS-10.0+-000000?style=flat&logo=apple&logoColor=white" alt="watchOS 10.0+">
  <img src="https://img.shields.io/badge/tvOS-17.0+-000000?style=flat&logo=apple&logoColor=white" alt="tvOS 17.0+">
  <img src="https://img.shields.io/badge/visionOS-1.0+-000000?style=flat&logo=apple&logoColor=white" alt="visionOS 1.0+">
</p>

## Learn More

<p align="center">
  <a href="https://academy.rudrank.com/product/on-device-ai" target="_blank">
    <img src="https://img.shields.io/badge/Book-Exploring%20On--Device%20AI-blue?style=for-the-badge&logo=book&logoColor=white" alt="Exploring On-Device AI Book">
  </a>
  <a href="https://academy.rudrank.com/product/ai-assisted-coding" target="_blank">
    <img src="https://img.shields.io/badge/Book-Exploring%20AI--Assisted%20Coding-blue?style=for-the-badge&logo=book&logoColor=white" alt="Exploring AI-Assisted Coding Book">
  </a>
</p>

Explore the following books to understand more about AI and iOS development:
- [Exploring On-Device AI for Apple Platforms Development](https://academy.rudrank.com/product/on-device-ai)
- [Exploring AI-Assisted Coding for iOS Development](https://academy.rudrank.com/product/ai-assisted-coding)

## Table of Contents

- [Features](#features)
- [Supported Platforms](#supported-platforms)
- [Installation](#installation)
  - [Swift Package Manager](#swift-package-manager)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Import VecturaKit](#import-vecturakit)
  - [Create Configuration and Initialize Database](#create-configuration-and-initialize-database)
  - [Add Documents](#add-documents)
  - [Search Documents](#search-documents)
  - [Document Management](#document-management)
  - [Database Information](#database-information)
  - [Custom Storage Provider](#custom-storage-provider)
  - [Custom Search Engine](#custom-search-engine)
- [MLX Integration](#mlx-integration)
  - [Import MLX Support](#import-mlx-support)
  - [Initialize Database with MLX](#initialize-database-with-mlx)
  - [Add Documents](#add-documents-1)
  - [Search Documents](#search-documents-1)
  - [Document Management](#document-management-1)
- [NaturalLanguage Integration](#naturallanguage-integration)
  - [Import NaturalLanguage Support](#import-naturallanguage-support)
  - [Initialize Database with NLContextualEmbedding](#initialize-database-with-nlcontextualembedding)
  - [Add Documents](#add-documents-2)
  - [Search Documents](#search-documents-2)
  - [Document Management](#document-management-2)
- [Command Line Interface](#command-line-interface)
  - [Swift CLI Tool (`vectura-cli`)](#swift-cli-tool-vectura-cli)
  - [MLX CLI Tool (`vectura-mlx-cli`)](#mlx-cli-tool-vectura-mlx-cli)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Features

-   **Model2Vec Support:** Uses the retrieval 32M parameter Model2Vec model as default for fast static embeddings.
-   **Auto-Dimension Detection:** Automatically detects embedding dimensions from models.
-   **On-Device Storage:** Stores and manages vector embeddings locally.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for relevant search results (`VecturaKit`).
-   **Pluggable Search Engines:** Implement custom search algorithms by conforming to the `VecturaSearchEngine` protocol.
-   **Batch Processing:** Indexes documents in parallel for faster data ingestion.
-   **Persistent Storage:** Automatically saves and loads document data, preserving the database state across app sessions.
-   **Configurable Search:** Customizes search behavior with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specifies a custom directory for database storage.
-   **Custom Storage Provider:** Implements custom storage backends (SQLite, Core Data, cloud storage) by conforming to the `VecturaStorage` protocol.
-   **Memory Management Strategies:** Choose between automatic, full-memory, or indexed modes to optimize performance for datasets ranging from thousands to millions of documents. [Learn more](Docs/INDEXED_STORAGE_GUIDE.md)
-   **MLX Support:** Uses Apple's MLX framework for accelerated embedding generation through `MLXEmbedder`.
-   **NaturalLanguage Support:** Uses Apple's NaturalLanguage framework for contextual embeddings with zero external dependencies through `NLContextualEmbedder`.
-   **CLI Tools:** Includes `vectura-cli` (Swift embeddings) and `vectura-mlx-cli` (MLX embeddings) for database management and testing.

## Supported Platforms

-   macOS 14.0 or later
-   iOS 17.0 or later
-   tvOS 17.0 or later
-   visionOS 1.0 or later
-   watchOS 10.0 or later

## Installation

Swift Package Manager handles the distribution of Swift code and comes built into the Swift compiler.

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "2.3.1"),
],
```

### Dependencies

VecturaKit uses the following Swift packages:

-   [swift-embeddings](https://github.com/jkrukowski/swift-embeddings): Used in `VecturaKit` for generating text embeddings using various models.
-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): Used for creating the command-line interface.
-   [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): Provides MLX-based embeddings and vector search capabilities, specifically for `VecturaMLXKit`.

**Note:** `VecturaNLKit` has no external dependencies beyond Apple's native NaturalLanguage framework.

## Quick Start

Get up and running with VecturaKit in minutes. Here is an example of adding and searching documents:

```swift
import VecturaKit

Task {
    do {
        let config = VecturaConfig(name: "my-db")
        let embedder = SwiftEmbedder(modelSource: .default)
        let vectorDB = try await VecturaKit(config: config, embedder: embedder)
        
        // Add documents
        let ids = try await vectorDB.addDocuments(texts: [
            "The quick brown fox jumps over the lazy dog",
            "Swift is a powerful programming language"
        ])
        
        // Search documents
        let results = try await vectorDB.search(query: "programming language", numResults: 5)
        print("Found \(results.count) results!")
    } catch {
        print("Error: \(error)")
    }
}
```

## Usage

### Import VecturaKit

```swift
import VecturaKit
```

### Create Configuration and Initialize Database

```swift
import Foundation
import VecturaKit

let config = VecturaConfig(
    name: "my-vector-db",
    directoryURL: nil,  // Optional custom storage location
    dimension: nil,     // Auto-detect dimension from embedder (recommended)
    searchOptions: VecturaConfig.SearchOptions(
        defaultNumResults: 10,
        minThreshold: 0.7,
        hybridWeight: 0.5,  // Balance between vector and text search
        k1: 1.2,           // BM25 parameters
        b: 0.75,
        bm25NormalizationFactor: 10.0
    )
)

// Create an embedder (SwiftEmbedder uses swift-embeddings library)
let embedder = SwiftEmbedder(modelSource: .default)
let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

**For large-scale datasets (100K+ documents):**

```swift
let config = VecturaConfig(
    name: "my-vector-db",
    memoryStrategy: .indexed(candidateMultiplier: 10)
)

let vectorDB = try await VecturaKit(config: config, embedder: embedder)
// Reduced memory footprint with on-demand document loading
```

> 💡 **Tip:** See the [Indexed Storage Guide](Docs/INDEXED_STORAGE_GUIDE.md) for detailed information on memory strategies and performance optimization for large-scale datasets.
>
> 📊 **Performance:** Check out the [Performance Test Results](Docs/TEST_RESULTS_SUMMARY.md) for detailed benchmarking data and recommendations. For documentation index, see [Docs/](Docs/).

### Add Documents

Single document:

```swift
let text = "Sample text to be embedded"
let documentId = try await vectorDB.addDocument(
    text: text,
    id: UUID()  // Optional, will be generated if not provided
)
```

Multiple documents in batch:

```swift
let texts = [
    "First document text",
    "Second document text",
    "Third document text"
]
let documentIds = try await vectorDB.addDocuments(
    texts: texts,
    ids: nil  // Optional array of UUIDs
)
```

### Search Documents

Search by text (hybrid search):

```swift
let results = try await vectorDB.search(
    query: "search query",
    numResults: 5,  // Optional
    threshold: 0.8   // Optional
)

for result in results {
    print("Document ID: \(result.id)")
    print("Text: \(result.text)")
    print("Similarity Score: \(result.score)")
    print("Created At: \(result.createdAt)")
}
```

Search by vector embedding:

```swift
// Using array literal
let results = try await vectorDB.search(
    query: [0.1, 0.2, 0.3, ...],  // Array literal matching config.dimension
    numResults: 5,  // Optional
    threshold: 0.8  // Optional
)

// Or explicitly use SearchQuery enum
let embedding: [Float] = getEmbedding()
let results = try await vectorDB.search(
    query: .vector(embedding),
    numResults: 5,
    threshold: 0.8
)
```

### Document Management

Update document:

```swift
try await vectorDB.updateDocument(
    id: documentId,
    newText: "Updated text"
)
```

Delete documents:

```swift
try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])
```

Reset database:

```swift
try await vectorDB.reset()
```

### Database Information

Get document count:

```swift
let count = await vectorDB.documentCount
print("Database contains \(count) documents")
```

### Custom Storage Provider

VecturaKit allows you to implement your own storage backend by conforming to the `VecturaStorage` protocol. This is useful for integrating with different storage systems like SQLite, Core Data, or cloud storage.

Define a custom storage provider:

```swift
import Foundation
import VecturaKit

final class MyCustomStorageProvider: VecturaStorage {
    private var documents: [UUID: VecturaDocument] = [:]

    func createStorageDirectoryIfNeeded() async throws {
        // Initialize your storage system
    }

    func loadDocuments() async throws -> [VecturaDocument] {
        // Load documents from your storage
        return Array(documents.values)
    }

    func saveDocument(_ document: VecturaDocument) async throws {
        // Save document to your storage
        documents[document.id] = document
    }

    func deleteDocument(withID id: UUID) async throws {
        // Delete document from your storage
        documents.removeValue(forKey: id)
    }

    func updateDocument(_ document: VecturaDocument) async throws {
        // Update document in your storage
        documents[document.id] = document
    }

    func getTotalDocumentCount() async throws -> Int {
        // Return total count (optional - protocol provides default implementation)
        return documents.count
    }
}
```

Use the custom storage provider:

```swift
let config = VecturaConfig(name: "my-db")
let customStorage = MyCustomStorageProvider()
let vectorDB = try await VecturaKit(
    config: config,
    storageProvider: customStorage
)

// Use vectorDB normally - all storage operations will use your custom provider
let documentId = try await vectorDB.addDocument(text: "Sample text")
```

### Custom Search Engine

VecturaKit supports custom search engine implementations by conforming to the `VecturaSearchEngine` protocol. This allows you to implement specialized search algorithms (pure vector, pure text, custom hybrid, or other ranking methods).

Define a custom search engine:

```swift
import Foundation
import VecturaKit

struct MyCustomSearchEngine: VecturaSearchEngine {

    func search(
        query: SearchQuery,
        storage: VecturaStorage,
        options: SearchOptions
    ) async throws -> [VecturaSearchResult] {
        // Load documents from storage
        let documents = try await storage.loadDocuments()

        // Implement your custom search logic
        // This example does a simple exact text match
        guard case .text(let queryText) = query else {
            return []
        }

        let results = documents.filter { doc in
            doc.text.lowercased().contains(queryText.lowercased())
        }.map { doc in
            VecturaSearchResult(
                id: doc.id,
                text: doc.text,
                score: 1.0,
                createdAt: doc.createdAt
            )
        }

        return Array(results.prefix(options.numResults))
    }

    func indexDocument(_ document: VecturaDocument) async throws {
        // Optional: Update your search engine's internal index
    }

    func removeDocument(id: UUID) async throws {
        // Optional: Remove from your search engine's internal index
    }
}
```

Use the custom search engine:

```swift
let config = VecturaConfig(name: "my-db")
let embedder = SwiftEmbedder(modelSource: .default)
let customEngine = MyCustomSearchEngine()

let vectorDB = try await VecturaKit(
    config: config,
    embedder: embedder,
    searchEngine: customEngine
)

// All searches will use your custom search engine
let results = try await vectorDB.search(query: "search query")
```

## MLX Integration

VecturaKit supports Apple's MLX framework through the `MLXEmbedder` for accelerated on-device machine learning performance.

### Import MLX Support

```swift
import VecturaKit
import VecturaMLXKit
import MLXEmbedders
```

### Initialize Database with MLX

```swift
let config = VecturaConfig(
  name: "my-mlx-vector-db",
  dimension: nil  // Auto-detect dimension from MLX embedder
)

// Create MLX embedder
let embedder = try await MLXEmbedder(configuration: .nomic_text_v1_5)
let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

### Add Documents

```swift
let texts = [
  "First document text",
  "Second document text",
  "Third document text"
]
let documentIds = try await vectorDB.addDocuments(texts: texts)
```

### Search Documents

```swift
let results = try await vectorDB.search(
    query: "search query",
    numResults: 5,      // Optional
    threshold: 0.8     // Optional
)

for result in results {
    print("Document ID: \(result.id)")
    print("Text: \(result.text)")
    print("Similarity Score: \(result.score)")
    print("Created At: \(result.createdAt)")
}
```

### Document Management

Update document:

```swift
try await vectorDB.updateDocument(
     id: documentId,
     newText: "Updated text"
 )
```

Delete documents:

```swift
try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])
```

Reset database:

```swift
try await vectorDB.reset()
```

## NaturalLanguage Integration

VecturaKit supports Apple's NaturalLanguage framework through the `NLContextualEmbedder` for contextual embeddings with zero external dependencies.

### Import NaturalLanguage Support

```swift
import VecturaKit
import VecturaNLKit
```

### Initialize Database with NLContextualEmbedding

```swift
let config = VecturaConfig(
  name: "my-nl-vector-db",
  dimension: nil  // Auto-detect dimension from NL embedder
)

// Create NLContextualEmbedder
let embedder = try await NLContextualEmbedder(
  language: .english
)
let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

**Available Options:**

```swift
// Initialize with specific language
let embedder = try await NLContextualEmbedder(
  language: .spanish
)

// Get model information
let modelInfo = await embedder.modelInfo
print("Language: \(modelInfo.language)")
print("Dimension: \(modelInfo.dimension ?? 0)")
```

### Add Documents

```swift
let texts = [
  "Natural language understanding is fascinating",
  "Swift makes iOS development enjoyable",
  "Machine learning on device preserves privacy"
]
let documentIds = try await vectorDB.addDocuments(texts: texts)
```

### Search Documents

```swift
let results = try await vectorDB.search(
    query: "iOS programming",
    numResults: 5,      // Optional
    threshold: 0.7     // Optional
)

for result in results {
    print("Document ID: \(result.id)")
    print("Text: \(result.text)")
    print("Similarity Score: \(result.score)")
    print("Created At: \(result.createdAt)")
}
```

### Document Management

Update document:

```swift
try await vectorDB.updateDocument(
     id: documentId,
     newText: "Updated text"
 )
```

Delete documents:

```swift
try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])
```

Reset database:

```swift
try await vectorDB.reset()
```

**Key Features:**

- **Zero External Dependencies:** Uses only Apple's native NaturalLanguage framework
- **Contextual Embeddings:** Considers surrounding context for more accurate semantic understanding
- **Privacy-First:** All processing happens on-device
- **Language Support:** Supports multiple languages (English, Spanish, French, German, Italian, Portuguese, and more)
- **Auto-Detection:** Automatically detects embedding dimensions

**Performance Characteristics:**

- **Speed:** Moderate (slower than Model2Vec, comparable to MLX)
- **Accuracy:** High contextual understanding for supported languages
- **Memory:** Efficient on-device processing
- **Use Cases:** Ideal for apps requiring semantic search without external dependencies

**Platform Requirements:**

- iOS 17.0+ / macOS 14.0+ / tvOS 17.0+ / visionOS 1.0+ / watchOS 10.0+
- NaturalLanguage framework (included with OS)

## Command Line Interface

VecturaKit includes command-line tools for database management with different embedding backends.

### Swift CLI Tool (`vectura-cli`)

```bash
# Add documents (dimension auto-detected from model)
vectura add "First document" "Second document" "Third document" \
  --db-name "my-vector-db"

# Search documents
vectura search "search query" \
  --db-name "my-vector-db" \
  --threshold 0.7 \
  --num-results 5

# Update document
vectura update <document-uuid> "Updated text content" \
  --db-name "my-vector-db"

# Delete documents
vectura delete <document-uuid-1> <document-uuid-2> \
  --db-name "my-vector-db"

# Reset database
vectura reset \
  --db-name "my-vector-db"

# Run demo with sample data
vectura mock \
  --db-name "my-vector-db" \
  --threshold 0.7 \
  --num-results 10
```

Common options for `vectura-cli`:

-   `--db-name, -d`: Database name (default: "vectura-cli-db")
-   `--dimension, -v`: Vector dimension (auto-detected by default)
-   `--threshold, -t`: Minimum similarity threshold (default: 0.7)
-   `--num-results, -n`: Number of results to return (default: 10)
-   `--model-id, -m`: Model ID for embeddings (default: "minishlab/potion-retrieval-32M")

### MLX CLI Tool (`vectura-mlx-cli`)

```bash
# Add documents
vectura-mlx add "First document" "Second document" "Third document" --db-name "my-mlx-vector-db"

# Search documents
vectura-mlx search "search query" --db-name "my-mlx-vector-db"  --threshold 0.7 --num-results 5

# Update document
vectura-mlx update <document-uuid> "Updated text content" --db-name "my-mlx-vector-db"

# Delete documents
vectura-mlx delete <document-uuid-1> <document-uuid-2> --db-name "my-mlx-vector-db"

# Reset database
vectura-mlx reset --db-name "my-mlx-vector-db"

# Run demo with sample data
vectura-mlx mock  --db-name "my-mlx-vector-db"
```

Options for `vectura-mlx-cli`:

-   `--db-name, -d`: Database name (default: "vectura-mlx-cli-db")
-   `--threshold, -t`: Minimum similarity threshold (default: no threshold)
-   `--num-results, -n`: Number of results to return (default: 10)

## License

VecturaKit is released under the MIT License. See the LICENSE file for more information. Copyright (c) 2025 Rudrank Riyam.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

### Development Setup

1. Clone the repository
2. Open `Package.swift` in Xcode or VS Code
3. Run tests to ensure everything works: `swift test`
4. Run performance benchmarks (optional): `swift test --filter BenchmarkSuite` — see [Performance Tests](Tests/PerformanceTests/)
5. Make your changes and test them

### Code Style

- Follow SwiftLint rules (run `swiftlint lint`)
- Use Swift 6.0+ features where appropriate
- Maintain backward compatibility when possible
- Document public APIs with DocC comments

## Support

- [Issues](https://github.com/rryam/VecturaKit/issues)
- [Discussions](https://github.com/rryam/VecturaKit/discussions)
- [Twitter](https://x.com/rudrankriyam)