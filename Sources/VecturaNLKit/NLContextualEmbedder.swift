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
public actor NLContextualEmbedder {

  private let embedding: NLEmbedding
  private let language: NLLanguage
  private var cachedDimension: Int?

  /// Initializes an NLContextualEmbedder with the specified language.
  ///
  /// - Parameters:
  ///   - language: The natural language for which to generate embeddings. Defaults to English.
  /// - Throws: `NLContextualEmbedderError.embeddingNotAvailable` if the embedding model
  ///           is not available for the specified language.
  ///
  /// ## Commonly Supported Languages
  /// The NaturalLanguage framework typically supports the following languages:
  /// - English, Spanish, French, German, Italian, Portuguese
  /// - Dutch, Swedish, Danish, Norwegian, Finnish
  /// - Russian, Polish, Turkish, Arabic, Hindi, Chinese, Japanese, Korean
  ///
  /// Language availability may vary by platform version and device.
  /// For the most up-to-date list, refer to Apple's NaturalLanguage framework documentation.
  public init(
    language: NLLanguage = .english
  ) async throws {
    self.language = language

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
  /// processes texts sequentially. While NLEmbedding is documented as thread-safe,
  /// it is not marked as Sendable by Apple, preventing safe concurrent access
  /// in Swift 6+ strict concurrency mode.
  ///
  /// For better performance with large batches, consider using other embedders
  /// like SwiftEmbedder or MLXEmbedder which support concurrent processing.
  ///
  /// - Parameter texts: The text strings to embed.
  /// - Returns: An array of embedding vectors, one for each input text.
  /// - Throws: `NLContextualEmbedderError.embeddingGenerationFailed` if any embedding fails.
  public func embed(texts: [String]) async throws -> [[Float]] {
    var results: [[Float]] = []
    results.reserveCapacity(texts.count)

    for (index, text) in texts.enumerated() {
      guard !text.isEmpty else {
        throw NLContextualEmbedderError.embeddingGenerationFailed(
          text: text,
          reason: "Text cannot be empty at index \(index)"
        )
      }

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
    // Returns nil if the embedding model is not available for the language or text
    guard let doubleVector = embedding.vector(for: text) else {
      throw NLContextualEmbedderError.embeddingGenerationFailed(
        text: text,
        reason: "NLEmbedding failed to generate vector. This may occur if the model is not downloaded or the text is incompatible with the language model (language: \(language.rawValue))"
      )
    }

    let floatVector = doubleVector.map { Float($0) }

    // Sanity check: This should never happen if doubleVector was non-empty,
    // but we verify to ensure data integrity
    guard !floatVector.isEmpty else {
      throw NLContextualEmbedderError.embeddingGenerationFailed(
        text: text,
        reason: "Unexpected error: Double-to-Float conversion resulted in empty vector (original length: \(doubleVector.count))"
      )
    }

    return floatVector
  }
}

// MARK: - Additional Information

extension NLContextualEmbedder {

  /// Returns information about the embedding model being used.
  public var modelInfo: ModelInfo {
    ModelInfo(
      language: language,
      dimension: cachedDimension
    )
  }

  /// Information about the NLEmbedding model.
  public struct ModelInfo: Sendable {
    /// The language of the embedding model.
    public let language: NLLanguage

    /// The embedding dimension, if already determined.
    public let dimension: Int?
  }
}
