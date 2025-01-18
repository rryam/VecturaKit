// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VecturaKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "VecturaKit",
            targets: ["VecturaKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.7")
    ],
    targets: [
        .target(
            name: "VecturaKit",
            dependencies: [
                .product(name: "Embeddings", package: "swift-embeddings")
            ]
        ),
        .testTarget(
            name: "VecturaKitTests",
            dependencies: ["VecturaKit"]
        ),
    ]
)
