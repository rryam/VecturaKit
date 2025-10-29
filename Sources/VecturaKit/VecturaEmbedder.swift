import Foundation

/// A protocol defining the requirements for an embedding provider.
///
/// Conforming types can generate vector embeddings from text using different underlying models
/// (e.g., swift-embeddings, MLX, OpenAI, etc.).
@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public protocol VecturaEmbedder: Sendable {

    /// The dimensionality of the embedding vectors produced by this embedder.
    var dimension: Int { get async throws }

    /// Generates embeddings for multiple texts in batch.
    ///
    /// - Parameter texts: The text strings to embed.
    /// - Returns: An array of embedding vectors, one for each input text.
    func embed(texts: [String]) async throws -> [[Float]]

    /// Generates an embedding for a single text.
    ///
    /// - Parameter text: The text string to embed.
    /// - Returns: The embedding vector for the input text.
    func embed(text: String) async throws -> [Float]
}

// MARK: - Default Implementation

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public extension VecturaEmbedder {

    /// Default implementation of single-text embedding using batch embedding.
    func embed(text: String) async throws -> [Float] {
        let embeddings = try await embed(texts: [text])
        guard let embedding = embeddings.first else {
            throw VecturaError.invalidInput("Failed to generate embedding for text")
        }
        return embedding
    }
}
