```markdown
# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device applications, enabling advanced user experiences through local vector storage and retrieval. Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** uses `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings). It also includes a version that leverages Apple's MLX framework for accelerated processing (VecturaMLXKit).

## Features

-   **On-Device Storage:** Store and manage vector embeddings directly on the device for enhanced privacy and reduced latency.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for more comprehensive and relevant search results.
-   **Batch Processing:** Efficiently add multiple documents in parallel for faster indexing.
-   **Persistent Storage:** Automatically saves and loads document data between app sessions.
-   **Configurable Search:** Customize search results with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specify a custom directory for database storage to suit specific app requirements.
-   **MLX Support:** Utilizes Apple's MLX framework for accelerated embedding generation and search capabilities (VecturaMLXKit).
-   **CLI Tool:** Includes a command-line interface for easy database management, testing and debugging.

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
    .package(url: "https://github.com/rryam/VecturaKit.git", branch: "main"),
],
```

### Dependencies

VecturaKit relies on the following Swift packages:

-   [swift-embeddings](https://github.com/jkrukowski/swift-embeddings): For generating text embeddings.
-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): For creating the command-line interface.
-   [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): For MLX based embeddings and vector search.

## Usage

### Core VecturaKit

1.  **Import VecturaKit**

    ```swift
    import VecturaKit
    ```

2.  **Create Configuration and Initialize Database**

    ```swift
    let config = VecturaConfig(
        name: "my-vector-db",
        directoryURL: nil,  // Optional custom storage location
        dimension: 384,     // Matches the default BERT model dimension
        searchOptions: VecturaConfig.SearchOptions(
            defaultNumResults: 10,
            minThreshold: 0.7,
            hybridWeight: 0.5,  // Balance between vector and text search
            k1: 1.2,           // BM25 parameters
            b: 0.75
        )
    )

    let vectorDB = try VecturaKit(config: config)
    ```

3.  **Add Documents**

    Single document:

    ```swift
    let text = "Sample text to be embedded"
    let documentId = try await vectorDB.addDocument(
        text: text,
        id: UUID(),  // Optional, will be generated if not provided
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional, this is the default
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
        ids: nil,  // Optional array of UUIDs
         model: .id("sentence-transformers/all-MiniLM-L6-v2") // Optional model
    )
    ```

4.  **Search Documents**

    Search by text (hybrid search):

    ```swift
    let results = try await vectorDB.search(
        query: "search query",
        numResults: 5,      // Optional
        threshold: 0.8,     // Optional
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional
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

5.  **Document Management**

    Update document:

    ```swift
    try await vectorDB.updateDocument(
        id: documentId,
        newText: "Updated text",
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional
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

### VecturaMLXKit (MLX Version)

VecturaMLXKit uses Apple's MLX framework for accelerated processing.

1.  **Import VecturaMLXKit**

    ```swift
    import VecturaMLXKit
    ```

2.  **Initialize Database**

    ```swift
    let config = VecturaConfig(
      name: "my-mlx-vector-db",
      dimension: 768 //  nomic_text_v1_5 model outputs 768-dimensional embeddings
    )
    let vectorDB = try await VecturaMLXKit(config: config)
    ```

3.  **Add Documents**

    ```swift
        let texts = [
            "First document text",
            "Second document text",
            "Third document text"
        ]
        let documentIds = try await vectorDB.addDocuments(texts: texts)
    ```

4.  **Search Documents**

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

5.  **Document Management**

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

VecturaKit comes with a built-in CLI tool for database operations:

```bash
# Add documents
vectura add "First document" "Second document" "Third document" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Search documents
vectura search "search query" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --threshold 0.7 \
  --num-results 5 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Update document
vectura update <document-uuid> "Updated text content" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Delete documents
vectura delete <document-uuid-1> <document-uuid-2> \
  --db-name "my-vector-db" \
  --dimension 384

# Reset database
vectura reset \
  --db-name "my-vector-db" \
  --dimension 384

# Run demo with sample data
vectura mock \
  --db-name "my-vector-db" \
  --dimension 384 \
  --threshold 0.7 \
  --num-results 10 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"
```

Common options:

-   `--db-name, -d`: Database name (default: "vectura-cli-db")
-   `--dimension, -v`: Vector dimension (default: 384)
-   `--threshold, -t`: Minimum similarity threshold (default: 0.7)
-   `--num-results, -n`: Number of results to return (default: 10)
-   `--model-id, -m`: Model ID for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")

**MLX CLI Tool**

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

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

VecturaKit is released under the MIT License. See the [LICENSE](LICENSE) file for more information.
```
