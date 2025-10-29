import CoreML
import Embeddings
import Foundation

/// An embedder implementation using swift-embeddings library (Bert and Model2Vec models).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public actor SwiftEmbedder: VecturaEmbedder {

    private let modelSource: VecturaModelSource
    private var bertModel: Bert.ModelBundle?
    private var model2vecModel: Model2Vec.ModelBundle?
    private var cachedDimension: Int?

    /// Initializes a SwiftEmbedder with the specified model source.
    ///
    /// - Parameter modelSource: The source from which to load the embedding model.
    public init(modelSource: VecturaModelSource = .default) {
        self.modelSource = modelSource
    }

    public var dimension: Int {
        get async throws {
            if let cached = cachedDimension {
                return cached
            }

            // Ensure model is loaded
            try await ensureModelLoaded()

            let dim: Int
            if let model2vec = model2vecModel {
                dim = model2vec.model.dimienstion
            } else if let bert = bertModel {
                // For BERT, we need to get dimension from a test encoding
                let testEmbedding = try bert.encode("test")
                dim = testEmbedding.shape.last ?? 0
            } else {
                throw VecturaError.invalidInput("No model loaded to detect dimension")
            }

            cachedDimension = dim
            return dim
        }
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        try await ensureModelLoaded()

        let embeddingsTensor: MLTensor
        if let model2vec = model2vecModel {
            embeddingsTensor = try model2vec.batchEncode(texts)
        } else if let bert = bertModel {
            embeddingsTensor = try bert.batchEncode(texts)
        } else {
            throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
        }

        let shape = embeddingsTensor.shape
        guard shape.count == 2 else {
            throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
        }

        let dimension = shape[1]
        let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(of: Float.self)
        let allScalars = embeddingShapedArray.scalars

        var result: [[Float]] = []
        result.reserveCapacity(texts.count)

        for i in 0..<texts.count {
            let startIndex = i * dimension
            let endIndex = startIndex + dimension
            let embeddingRow = Array(allScalars[startIndex..<endIndex])
            result.append(embeddingRow)
        }

        return result
    }

    public func embed(text: String) async throws -> [Float] {
        try await ensureModelLoaded()

        let embeddingTensor: MLTensor
        if let model2vec = model2vecModel {
            embeddingTensor = try model2vec.encode(text)
        } else if let bert = bertModel {
            embeddingTensor = try bert.encode(text)
        } else {
            throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
        }

        let embeddingShapedArray = await embeddingTensor.cast(to: Float.self).shapedArray(of: Float.self)
        return embeddingShapedArray.scalars
    }

    // MARK: - Private

    private func ensureModelLoaded() async throws {
        guard bertModel == nil && model2vecModel == nil else {
            return
        }

        if isModel2VecModel(modelSource) {
            model2vecModel = try await Model2Vec.loadModelBundle(from: modelSource)
        } else {
            bertModel = try await Bert.loadModelBundle(from: modelSource)
        }
    }

    private func isModel2VecModel(_ source: VecturaModelSource) -> Bool {
        let modelId = source.description
        return modelId.contains("minishlab") ||
               modelId.contains("potion") ||
               modelId.contains("model2vec") ||
               modelId.contains("M2V")
    }
}

// MARK: - Model Loading Extensions

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    static func loadModelBundle(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
        switch source {
        case .id(let modelId):
            try await loadModelBundle(from: modelId)
        case .folder(let url):
            try await loadModelBundle(from: url)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Model2Vec {
    static func loadModelBundle(from source: VecturaModelSource) async throws -> Model2Vec.ModelBundle {
        switch source {
        case .id(let modelId):
            try await loadModelBundle(from: modelId)
        case .folder(let url):
            try await loadModelBundle(from: url)
        }
    }
}
