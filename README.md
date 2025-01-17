# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps, enabling user experiences through local vector storage and retrieval. Inspired by [Dripfarm’s SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** leverages MLXEmbeddings.

## Features
- On-Device Storage: Maintain data privacy and reduce latency by storing vectors directly on the device.
- MLXEmbeddings Support: Utilize MLXEmbeddings for accurate and meaningful vector representations.

## Installation

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rudrankriyam/VecturaKit.git", brach: "main"),
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

```swift
// TO-DO
let text = "Sample text to be embedded"
let embedding = MLXArray([...]) // Your embedding vector here
let documentId = try await vectorDB.addDocument(
text: text,
embedding: embedding
)
```

5. Perform a Search

```swift
let queryEmbedding = MLXArray([...])

let results = try await vectorDB.search(
query: queryEmbedding,
numResults: 5,
threshold: 0.8
)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

VecturaKit is released under the MIT License. See the LICENSE file for more information.
