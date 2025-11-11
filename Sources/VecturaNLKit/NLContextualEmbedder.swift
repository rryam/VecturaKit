import Foundation
import NaturalLanguage
import VecturaKit

/// An embedder implementation using Apple's NaturalLanguage framework for contextual embeddings.
///
/// This embedder leverages NLContextualEmbedding to generate contextual word embeddings that
/// consider the surrounding text context, providing semantically richer representations compared
/// to static embeddings.
///
/// ## Features
/// - Context-aware embeddings using Apple's on-device models
/// - Language-specific model support
/// - Full privacy with on-device processing
/// - No external dependencies beyond Apple frameworks
///
/// ## Example Usage
/// ```swift
/// let embedder = try await NLContextualEmbedder(language: .english)
/// let embedding = try await embedder.embed(text: "Swift is a powerful language")
/// ```
///
/// ## Availability
/// This embedder requires iOS 17.0+, macOS 14.0+, or equivalent platform versions
/// where NLEmbedding is available.
///
@available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public actor NLContextualEmbedder {

  private let embedding: NLEmbedding
  private let language: NLLanguage
  private let revision: Int?
  private var cachedDimension: Int?

  /// Initializes an NLContextualEmbedder with the specified language.
  ///
  /// - Parameters:
  ///   - language: The natural language for which to generate embeddings. Defaults to English.
  /// - Throws: `NLContextualEmbedderError.embeddingNotAvailable` if the embedding model
  ///           is not available for the specified language.
  public init(
    language: NLLanguage = .english
  ) async throws {
    self.language = language
    self.revision = nil

    // Use sentenceEmbedding for holistic semantic understanding of full sentences
    guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else {
      throw NLContextualEmbedderError.embeddingNotAvailable(
        language: language.rawValue
      )
    }

    self.embedding = embedding
  }
}

// MARK: - VecturaEmbedder

@available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
extension NLContextualEmbedder: VecturaEmbedder {

  /// The dimensionality of the embedding vectors produced by this embedder.
  ///
  /// This value is cached after first detection to avoid repeated computation.
  /// The dimension is determined by generating a test embedding.
  ///
  /// - Returns: The embedding dimension (typically 768 for most NLEmbedding models).
  /// - Throws: An error if the dimension cannot be determined.
  public var dimension: Int {
    get async throws {
      if let cached = cachedDimension {
        return cached
      }

      // Generate a test embedding to determine the dimension
      let testVector = try await embed(text: "test")
      let dim = testVector.count
      cachedDimension = dim
      return dim
    }
  }

  /// Generates embeddings for multiple texts in batch.
  ///
  /// Note: NLEmbedding does not provide a native batch API, so this method
  /// processes texts sequentially. For better performance with large batches,
  /// consider using other embedders like SwiftEmbedder or MLXEmbedder.
  ///
  /// - Parameter texts: The text strings to embed.
  /// - Returns: An array of embedding vectors, one for each input text.
  /// - Throws: `NLContextualEmbedderError.embeddingGenerationFailed` if any embedding fails.
  public func embed(texts: [String]) async throws -> [[Float]] {
    var results: [[Float]] = []
    results.reserveCapacity(texts.count)

    for text in texts {
      let vector = try await embed(text: text)
      results.append(vector)
    }

    return results
  }

  /// Generates an embedding for a single text.
  ///
  /// This method uses NLEmbedding's contextual word embedding to generate a
  /// sentence-level representation that captures semantic meaning based on context.
  ///
  /// - Parameter text: The text string to embed.
  /// - Returns: The embedding vector for the input text.
  /// - Throws: `NLContextualEmbedderError.embeddingGenerationFailed` if embedding generation fails.
  public func embed(text: String) async throws -> [Float] {
    guard !text.isEmpty else {
      throw NLContextualEmbedderError.embeddingGenerationFailed(
        text: text,
        reason: "Text cannot be empty"
      )
    }

    // NLEmbedding.vector(for:) returns [Double], VecturaEmbedder requires [Float]
    guard let doubleVector = embedding.vector(for: text) else {
      throw NLContextualEmbedderError.embeddingGenerationFailed(
        text: text,
        reason: "Failed to generate embedding vector"
      )
    }

    let floatVector = doubleVector.map { Float($0) }

    guard !floatVector.isEmpty else {
      throw NLContextualEmbedderError.embeddingGenerationFailed(
        text: text,
        reason: "Generated embedding is empty"
      )
    }

    return floatVector
  }
}

// MARK: - Additional Information

@available(iOS 17.0, macOS 14.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
extension NLContextualEmbedder {

  /// Returns information about the embedding model being used.
  public var modelInfo: ModelInfo {
    ModelInfo(
      language: language,
      revision: revision,
      dimension: cachedDimension
    )
  }

  /// Information about the NLEmbedding model.
  public struct ModelInfo: Sendable {
    /// The language of the embedding model.
    public let language: NLLanguage

    /// The model revision, if specified.
    public let revision: Int?

    /// The embedding dimension, if already determined.
    public let dimension: Int?
  }
}
