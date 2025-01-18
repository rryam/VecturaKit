# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps, enabling user experiences through local vector storage and retrieval. Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** uses `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings).

## Features
- On-Device Storage: Maintain data privacy and reduce latency by storing vectors directly on the device.
- MLXEmbeddings Support: Utilize MLXEmbeddings for accurate and meaningful vector representations.
- Batch Processing: Efficiently add multiple documents in parallel.
- Normalized Vectors: Pre-computed normalized embeddings for faster similarity search.

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

2. Create Configuration

```swift
let config = VecturaConfig(
    name: "my-vector-db",
    dimension: 384, // Set this to match your embedding dimension
    searchOptions: VecturaConfig.SearchOptions(
        defaultNumResults: 10,
        minThreshold: 0.7
    )
)
```

3. Initialize the Database

```swift
let vectorDB = try VecturaKit(config: config)
```

4. Add Documents

Single document:
```swift
let text = "Sample text to be embedded"
let documentId = try await vectorDB.addDocument(
    text: text,
    modelConfig: .nomic_text_v1_5  // Optional, this is the default
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
    modelConfig: .nomic_text_v1_5  // Optional, this is the default
)
```

5. Perform a Search

```swift
let results = try await vectorDB.search(
    query: queryText,
    numResults: 5,  // Optional, defaults to config.searchOptions.defaultNumResults
    threshold: 0.8  // Optional, defaults to config.searchOptions.minThreshold
)

for result in results {
    print("Document ID: \(result.id)")
    print("Text: \(result.text)")
    print("Similarity Score: \(result.score)")
}
```

6. Reset Database (Optional)

```swift
try await vectorDB.reset()
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

VecturaKit is released under the MIT License. See the LICENSE file for more information.
