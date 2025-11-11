import Foundation

/// Errors specific to NLContextualEmbedder operations.
public enum NLContextualEmbedderError: Error {

  /// Thrown when the contextual embedding model is not available for the requested language.
  case embeddingNotAvailable(language: String)

  /// Thrown when the embedding generation fails.
  case embeddingGenerationFailed(text: String, reason: String)
}

// MARK: - LocalizedError

extension NLContextualEmbedderError: LocalizedError {

  public var errorDescription: String? {
    switch self {
    case .embeddingNotAvailable(let language):
      "Contextual embedding not available for language: \(language)"
    case .embeddingGenerationFailed(let text, let reason):
      "Failed to generate embedding for text '\(text.prefix(50))...': \(reason)"
    }
  }
}
