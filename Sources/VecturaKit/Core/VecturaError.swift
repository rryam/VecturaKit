import Foundation

/// Errors that can occur when using VecturaKit.
public enum VecturaError {

  /// Thrown when vector dimensions don't match.
  case dimensionMismatch(expected: Int, got: Int)

  /// Thrown when loading data fails.
  case loadFailed(String)

  /// Thrown when input validation fails.
  case invalidInput(String)

  /// Thrown when attempting to access a document that doesn't exist.
  case documentNotFound(UUID)
}

// MARK: - LocalizedError

extension VecturaError: LocalizedError {

  public var errorDescription: String? {
    switch self {
    case .dimensionMismatch(let expected, let got):
      "Vector dimension mismatch. Expected \(expected) but got \(got)."
    case .loadFailed(let reason):
      "Failed to load data: \(reason)"
    case .invalidInput(let reason):
      "Invalid input: \(reason)"
    case .documentNotFound(let id):
      "Document with ID '\(id)' not found."
    }
  }
}
