import Foundation

/// Specifies where to obtain the resources for an embedding model.
public enum VecturaModelSource: Sendable {

  /// The type of embedding model architecture.
  public enum ModelType: Sendable {
    /// Standard BERT-based embedding models
    case bert
    /// Model2Vec distilled models (e.g., potion, minishlab models)
    case model2vec
  }

  /// Automatically fetch the model from a remote repository based on its id.
  ///
  /// - Parameters:
  ///   - id: The identifier of the model to fetch from the remote repository.
  ///   - type: Optional explicit model type. If nil, type will be inferred from model ID.
  case id(_ id: String, type: ModelType? = nil)

  /// Load a local model from the specified directory URL.
  ///
  /// - Parameters:
  ///   - url: The local directory URL containing the model files.
  ///   - type: Optional explicit model type. If nil, type will be inferred from directory name.
  case folder(_ url: URL, type: ModelType? = nil)
}

public extension VecturaModelSource {

  /// The default model identifier when not otherwise specified.
  static let defaultModelId = "minishlab/potion-retrieval-32M"

  /// The default model when not otherwise specified.
  static let `default` = Self.id(Self.defaultModelId)
}

// MARK: - CustomStringConvertible

extension VecturaModelSource: CustomStringConvertible {

  /// A human-readable description of the model source.
  ///
  /// For `.id` cases, returns the model identifier.
  /// For `.folder` cases, returns the file path.
  public var description: String {
    switch self {
    case .id(let id, _): id
    case .folder(let url, _): url.path(percentEncoded: false)
    }
  }
}
