// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VecturaKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "VecturaKit",
            targets: ["VecturaKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main")
    ],
    targets: [
        .target(
            name: "VecturaKit",
            dependencies: [
                .product(name: "MLXEmbedders", package: "mlx-swift-examples")
            ]
        ),
        .testTarget(
            name: "VecturaKitTests",
            dependencies: ["VecturaKit"]
        ),
    ]
)
