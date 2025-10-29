# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps through local vector storage and retrieval. 

Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** uses `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings) for generating and managing embeddings. It features **Model2Vec** support with the 32M parameter model as default for fast static embeddings.

The framework offers `VecturaKit` as the core vector database with pluggable embedding providers. Use `SwiftEmbedder` for `swift-embeddings` integration or `MLXEmbedder` for Apple's MLX framework acceleration.

It also includes CLI tools (`vectura-cli` and `vectura-mlx-cli`) for easily trying out the package.

## Learn More

Explore the following books to understand more about AI and iOS development:
- [Exploring On-Device AI for Apple Platforms Development](https://academy.rudrank.com/product/on-device-ai)
- [Exploring AI-Assisted Coding for iOS Development](https://academy.rudrank.com/product/ai-assisted-coding)

## Features

-   **Model2Vec Support:** Uses the retrieval 32M parameter Model2Vec model as default for fast static embeddings.
-   **Auto-Dimension Detection:** Automatically detects embedding dimensions from models.
-   **On-Device Storage:** Stores and manages vector embeddings locally.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for relevant search results (`VecturaKit`).
-   **Batch Processing:** Indexes documents in parallel for faster data ingestion.
-   **Persistent Storage:** Automatically saves and loads document data, preserving the database state across app sessions.
-   **Configurable Search:** Customizes search behavior with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specifies a custom directory for database storage.
-   **Custom Storage Provider:** Implements custom storage backends (SQLite, Core Data, cloud storage) by conforming to the `VecturaStorage` protocol.
-   **MLX Support:** Uses Apple's MLX framework for accelerated embedding generation through `MLXEmbedder`.
-   **CLI Tools:** Includes `vectura-cli` (Swift embeddings) and `vectura-mlx-cli` (MLX embeddings) for database management and testing.

## Supported Platforms

-   macOS 14.0 or later
-   iOS 17.0 or later
-   tvOS 17.0 or later
-   visionOS 1.0 or later
-   watchOS 10.0 or later

## Installation

### Swift Package Manager

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "1.1.0"),
],
```

### Dependencies

VecturaKit uses the following Swift packages:

-   [swift-embeddings](https://github.com/jkrukowski/swift-embeddings): Used in `VecturaKit` for generating text embeddings using various models.
-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): Used for creating the command-line interface.
-   [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): Provides MLX-based embeddings and vector search capabilities, specifically for `VecturaMLXKit`.

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
        b: 0.75
    )
)

// Create an embedder (SwiftEmbedder uses swift-embeddings library)
let embedder = SwiftEmbedder(modelSource: .default)
let vectorDB = try await VecturaKit(config: config, embedder: embedder)
```

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
let results = try await vectorDB.search(
    query: embeddingArray,  // [Float] matching config.dimension
    numResults: 5,  // Optional
    threshold: 0.8  // Optional
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
