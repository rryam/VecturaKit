// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaKit",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .visionOS(.v1),
    .watchOS(.v10),
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
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.21"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
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
    .executableTarget(
      name: "TestExamples",
      dependencies: ["VecturaKit"]
    ),
    .executableTarget(
      name: "TestMLXExamples",
      dependencies: ["VecturaMLXKit"]
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
