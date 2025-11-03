import Foundation

/// Specifies where to obtain the resources for an embedding model.
public enum VecturaModelSource: Sendable {

  /// Automatically fetch the model from a remote repository based on its id.
  ///
  /// - Parameter id: The identifier of the model to fetch from the remote repository.
  case id(_ id: String)
  /// Load a local model from the specified directory URL.
  ///
  /// - Parameter url: The local directory URL containing the model files.
  case folder(_ url: URL)
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
    case .id(let id): id
    case .folder(let url): url.path(percentEncoded: false)
    }
  }
}
