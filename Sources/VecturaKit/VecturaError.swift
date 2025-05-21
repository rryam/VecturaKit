import Foundation

/// Errors that can occur when using VecturaKit.
public enum VecturaError: LocalizedError {
    /// Thrown when attempting to create a collection that already exists.
    case collectionAlreadyExists(String)

    /// Thrown when attempting to access a collection that doesn't exist.
    case collectionNotFound(String)

    /// Thrown when vector dimensions don't match.
    case dimensionMismatch(expected: Int, got: Int)

    /// Thrown when loading collection data fails.
    case loadFailed(String)

    /// Thrown when input validation fails.
    case invalidInput(String)

    /// Thrown when VecturaKit initialization fails.
    case initializationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .collectionAlreadyExists(let name):
            "A collection named '\(name)' already exists."
        case .collectionNotFound(let name):
            "Collection '\(name)' not found."
        case .dimensionMismatch(let expected, let got):
            "Vector dimension mismatch. Expected \(expected) but got \(got)."
        case .loadFailed(let reason):
            "Failed to load collection: \(reason)"
        case .invalidInput(let reason):
            "Invalid input: \(reason)"
        case .initializationFailed(let reason):
            "Initialization failed: \(reason)"
        }
    }
}
