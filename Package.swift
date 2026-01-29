// swift-tools-version: 6.1
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
    .library(
      name: "VecturaNLKit",
      targets: ["VecturaNLKit"]
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
  traits: [
    .trait(
      name: "MLX",
      description: "Enable MLX-based embeddings for GPU-accelerated inference"
    ),
  ],
  dependencies: [
    // Always included - lightweight dependencies
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.21"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),

    // MLX dependency - only loaded when MLX trait is enabled via target conditions
    .package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.3"),
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
        .product(
          name: "MLXEmbedders",
          package: "mlx-swift-lm",
          condition: .when(traits: ["MLX"])
        ),
      ]
    ),
    .target(
      name: "VecturaNLKit",
      dependencies: [
        "VecturaKit"
      ]
    ),
    .executableTarget(
      name: "VecturaCLI",
      dependencies: [
        "VecturaKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
      resources: [
        .copy("Resources/mock_documents.json")
      ]
    ),
    .executableTarget(
      name: "VecturaMLXCLI",
      dependencies: [
        "VecturaKit",
        .target(name: "VecturaMLXKit", condition: .when(traits: ["MLX"])),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .executableTarget(
      name: "TestExamples",
      dependencies: ["VecturaKit"]
    ),
    .executableTarget(
      name: "TestMLXExamples",
      dependencies: [
        .target(name: "VecturaMLXKit", condition: .when(traits: ["MLX"]))
      ]
    ),
    .executableTarget(
      name: "TestNLExamples",
      dependencies: ["VecturaNLKit"]
    ),
    .testTarget(
      name: "VecturaKitTests",
      dependencies: ["VecturaKit"]
    ),
    .testTarget(
      name: "VecturaMLXKitTests",
      dependencies: [
        .target(name: "VecturaMLXKit", condition: .when(traits: ["MLX"]))
      ]
    ),
    .testTarget(
      name: "VecturaNLKitTests",
      dependencies: ["VecturaNLKit"]
    ),
    .testTarget(
      name: "PerformanceTests",
      dependencies: ["VecturaKit"],
      resources: [
        .copy("README.md"),
        .copy("ArchivedResults"),
        .copy("TestData"),
      ]
    ),
  ]
)
