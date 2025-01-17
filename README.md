# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device apps, enabling user experiences through local vector storage and retrieval. Inspired by [Dripfarm’s SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** leverages MLXEmbeddings.

## Features
	•	On-Device Storage: Maintain data privacy and reduce latency by storing vectors directly on the device.
	•	MLXEmbeddings Support: Utilize MLXEmbeddings for accurate and meaningful vector representations.

## Installation

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

dependencies: [
    .package(url: "https://github.com/rudrankriyam/VecturaKit.git", brach: "main"),
],

Replace yourusername with your GitHub username.

## Usage

1. Import VecturaKit

```swift
import VecturaKit
```

2. Initialize the Database

```swift
let vecturaDB = VecturaDatabase()
```

3. Add Documents

```swift
// TO-DO
let document = "Sample text to be embedded."

4. Perform a Search

```swift
// TO-DO
let query = "Relevant search query."
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

VecturaKit is released under the MIT License. See the LICENSE file for more information.
