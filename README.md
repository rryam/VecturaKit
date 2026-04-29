# VecturaKit


VecturaKit is a Swift-based vector database designed for on-device apps through local vector storage and retrieval.

Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** provides local vector storage, indexing, and hybrid search with pluggable embedding providers.

The framework offers `VecturaKit` as the core vector database with pluggable embedding providers. Use `OpenAICompatibleEmbedder` for hosted or local `/v1/embeddings` providers, `NLContextualEmbedder` for Apple's NaturalLanguage framework with zero external dependencies, [`SwiftEmbedder`](https://github.com/rryam/VecturaEmbeddingsKit) for `swift-embeddings` models, or [`MLXEmbedder`](https://github.com/rryam/VecturaMLXKit) for Apple's MLX framework acceleration.

It also includes `vectura-oai-cli` for trying OpenAI-compatible embedding providers from the command line.

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
- [OpenAI-Compatible Integration](#openai-compatible-integration)
  - [Import OpenAI-Compatible Support](#import-openai-compatible-support)
  - [Initialize Database with OpenAI-Compatible Embeddings](#initialize-database-with-openai-compatible-embeddings)
- [NaturalLanguage Integration](#naturallanguage-integration)
  - [Import NaturalLanguage Support](#import-naturallanguage-support)
  - [Initialize Database with NLContextualEmbedding](#initialize-database-with-nlcontextualembedding)
  - [Add Documents](#nl-add-documents)
  - [Search Documents](#nl-search-documents)
  - [Document Management](#nl-document-management)
- [Command Line Interface](#command-line-interface)
  - [OpenAI-Compatible CLI Tool (`vectura-oai-cli`)](#openai-compatible-cli-tool-vectura-oai-cli)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Features

-   **Pluggable Embeddings:** Bring any provider that conforms to `VecturaEmbedder`.
-   **Auto-Dimension Detection:** Automatically detects embedding dimensions from the configured embedder.
-   **On-Device Storage:** Stores and manages vector embeddings locally.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for relevant search results (`VecturaKit`).
-   **Pluggable Search Engines:** Implement custom search algorithms by conforming to the `VecturaSearchEngine` protocol.
-   **Batch Processing:** Indexes documents in parallel for faster data ingestion.
-   **Persistent Storage:** Automatically saves and loads document data, preserving the database state across app sessions.
-   **Configurable Search:** Customizes search behavior with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specifies a custom directory for database storage.
-   **Custom Storage Provider:** Implements custom storage backends (SQLite, Core Data, cloud storage) by conforming to the `VecturaStorage` protocol.
-   **Memory Management Strategies:** Choose between automatic, full-memory, or indexed modes to optimize performance for datasets ranging from thousands to millions of documents. [Learn more](Docs/INDEXED_STORAGE_GUIDE.md)
-   **Swift Embeddings Support:** Model2Vec, StaticEmbeddings, NomicBERT, ModernBERT, RoBERTa, XLM-RoBERTa, and BERT support is available via the separate [VecturaEmbeddingsKit](https://github.com/rryam/VecturaEmbeddingsKit) package.
-   **MLX Support:** GPU-accelerated embedding generation available via the separate [VecturaMLXKit](https://github.com/rryam/VecturaMLXKit) package.
-   **OpenAI-Compatible Support:** Connects to OpenAI-compatible `/v1/embeddings` endpoints exposed by local servers and hosted providers through `OpenAICompatibleEmbedder`.
-   **NaturalLanguage Support:** Uses Apple's NaturalLanguage framework for contextual embeddings with zero external dependencies through `NLContextualEmbedder`.
-   **CLI Tools:** Includes `vectura-oai-cli` for OpenAI-compatible database management and testing.

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
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "3.0.0"),
],
```

Then add the products you want to your target:

```swift
target(
    name: "MyApp",
    dependencies: [
        .product(name: "VecturaKit", package: "VecturaKit"),
        .product(name: "VecturaOAIKit", package: "VecturaKit"),
    ]
)
```

For MLX support, also add the separate [VecturaMLXKit](https://github.com/rryam/VecturaMLXKit) package:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "3.0.0"),
    .package(url: "https://github.com/rryam/VecturaMLXKit.git", from: "1.0.0"),
],
```

### Dependencies

VecturaKit uses the following Swift packages:

-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): Used for creating the command-line interface.

**Note:** `VecturaNLKit` and `VecturaOAIKit` are shipped from this package. For `swift-embeddings`-based models, see [VecturaEmbeddingsKit](https://github.com/rryam/VecturaEmbeddingsKit). For MLX-based embeddings, see [VecturaMLXKit](https://github.com/rryam/VecturaMLXKit).

## Quick Start

Get up and running with VecturaKit in minutes. Here is an example of adding and searching documents:

```swift
import VecturaKit
import VecturaNLKit

Task {
    do {
        let config = VecturaConfig(name: "my-db")
        let embedder = try await NLContextualEmbedder(language: .english)
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
import VecturaNLKit

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

// Create an embedder
let embedder = try await NLContextualEmbedder(language: .english)
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

Fetch a single document by ID:

```swift
let document = try await vectorDB.getDocument(id: documentId)
print(document?.text ?? "Document not found")
```

Check whether a document exists:

```swift
let exists = try await vectorDB.documentExists(id: documentId)
print("Exists: \(exists)")
```

Load all persisted documents:

```swift
let documents = try await vectorDB.getAllDocuments()
print("Loaded \(documents.count) documents")
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

`VecturaStorage` also provides default `getDocument(id:)` and `documentExists(id:)`
implementations. Custom providers can override them for more efficient single-document
lookups when their backend supports it.

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
let embedder = try await NLContextualEmbedder(language: .english)
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

For GPU-accelerated embeddings using Apple's MLX framework, see the separate [VecturaMLXKit](https://github.com/rryam/VecturaMLXKit) package. It provides `MLXEmbedder`, a drop-in `VecturaEmbedder` implementation, plus the `vectura-mlx-cli` command-line tool.

```swift
// Add both packages to your dependencies:
.package(url: "https://github.com/rryam/VecturaKit.git", from: "3.0.0"),
.package(url: "https://github.com/rryam/VecturaMLXKit.git", from: "1.0.0"),
```

## OpenAI-Compatible Integration

VecturaKit supports OpenAI-compatible embedding APIs through `OpenAICompatibleEmbedder`. This is useful for local servers such as Ollama, LM Studio, llama.cpp-compatible servers, vLLM, and hosted providers that expose `/v1/embeddings`.

### Import OpenAI-Compatible Support

```swift
import VecturaKit
import VecturaOAIKit
```

### Initialize Database with OpenAI-Compatible Embeddings

```swift
let config = VecturaConfig(
  name: "my-oai-vector-db",
  dimension: nil
)

let embedder = OpenAICompatibleEmbedder(
  baseURL: "http://localhost:1234/v1",
  model: "text-embedding-model",
  apiKey: nil,
  timeoutInterval: 120,
  retryAttempts: 2,
  retryBaseDelaySeconds: 1
)

let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

The embedder:

- Sends batched requests to `POST <baseURL>/embeddings`
- Adds `Authorization: Bearer ...` when an API key is configured
- Retries HTTP 429 responses and honors `Retry-After` when present
- Detects embedding dimension automatically on first use

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
if let dimension = modelInfo.dimension {
  print("Dimension: \(dimension)")
} else {
  print("Dimension: Not yet determined")
}
```

### <a name="nl-add-documents"></a>Add Documents

```swift
let texts = [
  "Natural language understanding is fascinating",
  "Swift makes iOS development enjoyable",
  "Machine learning on device preserves privacy"
]
let documentIds = try await vectorDB.addDocuments(texts: texts)
```

### <a name="nl-search-documents"></a>Search Documents

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

### <a name="nl-document-management"></a>Document Management

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

- **Speed:** Moderate (slower than Model2Vec, comparable to [MLX](https://github.com/rryam/VecturaMLXKit))
- **Accuracy:** High contextual understanding for supported languages
- **Memory:** Efficient on-device processing
- **Use Cases:** Ideal for apps requiring semantic search without external dependencies

**Platform Requirements:**

- iOS 17.0+ / macOS 14.0+ / tvOS 17.0+ / visionOS 1.0+ / watchOS 10.0+
- NaturalLanguage framework (included with OS)

## Command Line Interface

VecturaKit includes an OpenAI-compatible CLI for database management.

### OpenAI-Compatible CLI Tool (`vectura-oai-cli`)

Set `VECTURA_OAI_BASE_URL` and `VECTURA_OAI_MODEL`, or pass `--base-url` and `--model` directly.

```bash
vectura-oai add "First document" "Second document" --model text-embedding-model
vectura-oai search "search query" --model text-embedding-model --threshold 0.7 --num-results 5
vectura-oai update <document-uuid> "Updated text" --model text-embedding-model
vectura-oai delete <document-uuid-1> <document-uuid-2> --model text-embedding-model
vectura-oai reset --model text-embedding-model
vectura-oai mock --model text-embedding-model
```

Common options for `vectura-oai-cli`:

-   `--db-name, -d`: Database name (default: `"vectura-oai-cli-db"`)
-   `--directory`: Database directory (optional)
-   `--base-url`: OpenAI-compatible base URL
-   `--model`: Embedding model identifier
-   `--api-key`: Optional API key
-   `--timeout`: Request timeout in seconds
-   `--retry-attempts`: HTTP 429 retry attempts
-   `--retry-base-delay`: Base retry delay in seconds
-   `--threshold, -t`: Minimum similarity threshold (default: `0.7`)
-   `--num-results, -n`: Number of results to return (default: `10`)

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

[![Star History Chart](https://api.star-history.com/svg?repos=rryam/VecturaKit&type=Date)](https://star-history.com/#rryam/VecturaKit&Date)
