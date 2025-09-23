import Foundation

/// Specifies where to obtain the resources for an embedding model.
public enum VecturaModelSource: Sendable, CustomStringConvertible {
    /// Automatically fetch the model from a remote repository based on its id.
    case id(_ id: String)
    /// Load a local model from the specified directory URL.
    case folder(_ url: URL)
}

public extension VecturaModelSource {
    /// The default model identifier when not otherwise specified.
    static let defaultModelId: String = "minishlab/potion-retrieval-32M"

    /// The default model when not otherwise specified.
    static let `default` = VecturaModelSource.id(VecturaModelSource.defaultModelId)
}

public extension VecturaModelSource {
    var description: String {
        switch self {
        case .id(let id): id
        case .folder(let url): url.path(percentEncoded: false)
        }
    }
}
