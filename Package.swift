// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaKit",
  platforms: [
    .macOS(.v15),
    .iOS(.v18),
    .tvOS(.v18),
    .visionOS(.v2),
    .watchOS(.v11),
  ],
  products: [
    .library(
      name: "VecturaKit",
      targets: ["VecturaKit"]
    ),
    .library(
      name: "VecturaNLKit",
      targets: ["VecturaNLKit"]
    ),
    .executable(
      name: "vectura-cli",
      targets: ["VecturaCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.26"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
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
      name: "TestExamples",
      dependencies: ["VecturaKit"]
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
