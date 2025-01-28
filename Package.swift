// swift-tools-version: 6.0
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
    .executable(
      name: "vectura-cli",
      targets: ["VecturaCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.10"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
  ],
  targets: [
    .target(
      name: "VecturaKit",
      dependencies: [
        .product(name: "Embeddings", package: "swift-embeddings")
      ]
    ),
    .executableTarget(
      name: "VecturaCLI",
      dependencies: [
        "VecturaKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "VecturaKitTests",
      dependencies: ["VecturaKit"]
    ),
  ]
)
