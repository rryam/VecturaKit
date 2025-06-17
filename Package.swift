// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaKit",
  platforms: [
    .macOS(.v15),  // VecturaKit requires macOS 15.0+ due to swift-embeddings dependencies
    .iOS(.v18),    // VecturaKit requires iOS 18.0+ due to swift-embeddings dependencies  
    .tvOS(.v18),   // VecturaKit requires tvOS 18.0+ due to swift-embeddings dependencies
    .visionOS(.v2), // VecturaKit requires visionOS 2.0+ due to swift-embeddings dependencies
    .watchOS(.v11), // VecturaKit requires watchOS 11.0+ due to swift-embeddings dependencies
  ],
  products: [
    .library(
      name: "VecturaKit",
      targets: ["VecturaKit"]
    ),
    .library(
      name: "VecturaMLXKit",
      targets: ["VecturaMLXKit"]
    ),
    .executable(
      name: "vectura-cli",
      targets: ["VecturaCLI"]
    ),
    .executable(
      name: "vectura-mlx-cli",
      targets: ["VecturaMLXCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.10"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
    // Pin to specific tag for security. Update when new stable versions are released.
    .package(url: "https://github.com/ml-explore/mlx-swift-examples/", exact: "2.25.4"),
  ],
  targets: [
    .target(
      name: "VecturaKit",
      dependencies: [
        .product(name: "Embeddings", package: "swift-embeddings")
      ],
      cSettings: [
        .define("ACCELERATE_NEW_LAPACK"),
        .define("ACCELERATE_LAPACK_ILP64"),
      ]
    ),
    .target(
      name: "VecturaMLXKit",
      dependencies: [
        "VecturaKit",
        .product(name: "MLXEmbedders", package: "mlx-swift-examples"),
      ]
    ),
    .executableTarget(
      name: "VecturaCLI",
      dependencies: [
        "VecturaKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .executableTarget(
      name: "VecturaMLXCLI",
      dependencies: [
        "VecturaMLXKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "VecturaKitTests",
      dependencies: ["VecturaKit"]
    ),
    .testTarget(
      name: "VecturaMLXKitTests",
      dependencies: ["VecturaMLXKit"]
    ),
  ]
)
