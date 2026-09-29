// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "HybridPromptCache",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "HybridPromptCache", targets: ["HybridPromptCache"]),
    ],
    dependencies: [
        // **`main`, not a release.** Resuming a Qwen3.5 cache means handing
        // `generateTokens` the model's carried `LMOutput.State`, and the
        // `state:` parameter arrived after 3.31.4, the latest tag. This moves
        // to a version as soon as upstream tags one that has it.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
    ],
    targets: [
        .target(name: "HybridPromptCache",
                dependencies: [.product(name: "MLXLMCommon", package: "mlx-swift-lm")]),
        .testTarget(name: "HybridPromptCacheTests",
                    dependencies: ["HybridPromptCache",
                                   .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                                   .product(name: "MLXLLM", package: "mlx-swift-lm"),
                                   .product(name: "MLXVLM", package: "mlx-swift-lm")]),
    ]
)
