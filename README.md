# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps, enabling user experiences through local vector storage and retrieval. Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** uses `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings).

## Features
- On-Device Storage: Maintain data privacy and reduce latency by storing vectors directly on the device.
- Batch Processing: Efficiently add multiple documents in parallel.
- Persistent Storage: Documents are automatically saved and loaded between sessions.
- Configurable Search: Customize search results with thresholds and result limits.

## Installation

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", branch: "main"),
],
```

## Usage

1. Import VecturaKit

```swift
import VecturaKit
```

2. Create Configuration and Initialize Database

```swift
let config = VecturaConfig(
    name: "my-vector-db",
    dimension: 384,  // Matches the default BERT model dimension
    searchOptions: VecturaConfig.SearchOptions(
        defaultNumResults: 10,
        minThreshold: 0.7
    )
)

let vectorDB = try VecturaKit(config: config)
```

3. Add Documents

Single document:
```swift
let text = "Sample text to be embedded"
let documentId = try await vectorDB.addDocument(
    text: text,
    id: UUID(),  // Optional, will be generated if not provided
    modelId: "sentence-transformers/all-MiniLM-L6-v2"  // Optional, this is the default
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
    modelId: "sentence-transformers/all-MiniLM-L6-v2"
)
```

4. Search Documents

Search by text:
```swift
let results = try await vectorDB.search(
    query: "search query",
    numResults: 5,  // Optional
    threshold: 0.8,  // Optional
    modelId: "sentence-transformers/all-MiniLM-L6-v2"  // Optional
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

5. Document Management

Update document:
```swift
try await vectorDB.updateDocument(
    id: documentId,
    newText: "Updated text",
    modelId: "sentence-transformers/all-MiniLM-L6-v2"  // Optional
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
- `--db-name, -d`: Database name (default: "vectura-cli-db")
- `--dimension, -v`: Vector dimension (default: 384)
- `--threshold, -t`: Minimum similarity threshold (default: 0.7)
- `--num-results, -n`: Number of results to return (default: 10)
- `--model-id, -m`: Model ID for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

VecturaKit is released under the MIT License. See the LICENSE file for more information.
