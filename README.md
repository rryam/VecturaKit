```markdown
# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device applications, enabling advanced user experiences through local vector storage and retrieval. Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** utilizes `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings) for generating and managing embeddings. The framework offers two primary modules: `VecturaKit`, which supports diverse embedding models via `swift-embeddings`, and `VecturaMLXKit`, which leverages Apple's MLX framework for accelerated processing. It also includes command-line interface tools (`vectura-cli` and `vectura-mlx-cli`) for interacting with the databases.

## Support

Love this project? Check out my books to explore more of AI and iOS development:
- [Exploring AI for iOS Development](https://academy.rudrank.com/product/ai)
- [Exploring AI-Assisted Coding for iOS Development](https://academy.rudrank.com/product/ai-assisted-coding)

Your support helps to keep this project growing!

## Key Features

-   **On-Device Storage:** Stores and manages vector embeddings locally, enhancing privacy and reducing latency.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for comprehensive and relevant search results (`VecturaKit`).
-   **Batch Processing:** Indexes documents in parallel for faster data ingestion.
-   **Persistent Storage:** Automatically saves and loads document data, preserving the database state across app sessions.
-   **Configurable Search:** Customizes search behavior with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specifies a custom directory for database storage.
-   **MLX Support:** Employs Apple's MLX framework for accelerated embedding generation and search operations (`VecturaMLXKit`).
-   **CLI Tool:** Includes a command-line interface (CLI) for database management, testing, and debugging for both `VecturaKit` and `VecturaMLXKit`.

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

-   [swift-embeddings](https://github.com/jkrukowski/swift-embeddings): Used in `VecturaKit` for generating text embeddings using various models.
-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): Used for creating the command-line interface.
-   [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): Provides MLX-based embeddings and vector search capabilities, specifically for `VecturaMLXKit`.

The project also has the following dependencies, as specified in `Package.resolved`:
-   [GzipSwift](https://github.com/1024jp/GzipSwift) version 6.0.1: Compression library.
-   [Jinja](https://github.com/johnmai-dev/Jinja) version 1.1.1: Templating engine.
-   [mlx-swift](https://github.com/ml-explore/mlx-swift) version 0.21.2: Apple's MLX framework for accelerated processing.
-   [swift-collections](https://github.com/apple/swift-collections.git) version 1.1.4: Apple's collections library.
-   [swift-numerics](https://github.com/apple/swift-numerics) version 1.0.2:  Numerics library for Swift.
-   [swift-safetensors](https://github.com/jkrukowski/swift-safetensors.git) version 0.0.7: A library for reading and writing Safe Tensors.
-   [swift-sentencepiece](https://github.com/jkrukowski/swift-sentencepiece) version 0.0.6:  SentencePiece implementation in Swift.
-   [swift-transformers](https://github.com/huggingface/swift-transformers) version 0.1.18: Hugging Face Transformers implementation in Swift.

## Usage

### Core VecturaKit

1.  **Import VecturaKit**

    ```swift
    import VecturaKit
    ```

2.  **Create Configuration and Initialize Database**

    ```swift
    import Foundation
    import VecturaKit

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

    let vectorDB = try await VecturaKit(config: config)
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

VecturaMLXKit harnesses Apple's MLX framework for accelerated processing, delivering optimized performance for on-device machine learning tasks.

1.  **Import VecturaMLXKit**

    ```swift
    import VecturaMLXKit
    ```

2.  **Initialize Database**

    ```swift
    import VecturaMLXKit
    import MLXEmbedders

    let config = VecturaConfig(
      name: "my-mlx-vector-db",
      dimension: 768 //  nomic_text_v1_5 model outputs 768-dimensional embeddings
    )
    let vectorDB = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
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

VecturaKit includes a command-line interface for both the standard and MLX versions, facilitating easy database management.

**Standard CLI Tool (`vectura-cli`)**

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

Common options for `vectura-cli`:

-   `--db-name, -d`: Database name (default: "vectura-cli-db")
-   `--dimension, -v`: Vector dimension (default: 384)
-   `--threshold, -t`: Minimum similarity threshold (default: 0.7)
-   `--num-results, -n`: Number of results to return (default: 10)
-   `--model-id, -m`: Model ID for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")

**MLX CLI Tool (`vectura-mlx-cli`)**

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

VecturaKit is released under the MIT License. See the [LICENSE](LICENSE) file for more information. Copyright (c) 2025 Rudrank Riyam.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

### Development

The project is structured as a Swift Package.  It includes the following key targets:

- `VecturaKit`: The core vector database library.
- `VecturaMLXKit`:  The MLX-accelerated version of the library.
- `vectura-cli`:  The command-line interface for `VecturaKit`.
- `vectura-mlx-cli`: The command-line interface for `VecturaMLXKit`.

To build and test the project, use the following commands:

```bash
swift build
swift test
```

### Continuous Integration

The project uses GitHub Actions for continuous integration.  The following workflows are defined:

- `build_and_test_vectura.yml`: Builds and tests the `VecturaKit` library and `vectura-cli` tool on macOS.
- `build_and_test_mlx.yml`: Builds and tests the `VecturaMLXKit` library on macOS.
- `claude-code-review.yml`:  Runs a code review on pull requests using the Claude AI model.  This workflow can be configured to provide feedback on code quality, potential bugs, performance considerations, and security concerns.  It requires a `CLAUDE_CODE_OAUTH_TOKEN` secret.
- `claude.yml`: Allows interaction with the Claude AI model through issue comments and pull request reviews.  It requires a `CLAUDE_CODE_OAUTH_TOKEN` secret.  It is triggered by mentioning `@claude` in a comment, review, issue, or pull request.
- `update-readme.yml`: Automatically updates the `README.md` file based on the codebase using Gemini AI. It runs on pushes to the `main` branch and creates a pull request with the updated README. Requires a `GEMINI_API_KEY` secret.

```

