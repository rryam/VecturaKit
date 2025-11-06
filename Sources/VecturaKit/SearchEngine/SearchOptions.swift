import Foundation

/// Search options for configuring search behavior
public struct SearchOptions: Sendable {

  /// Maximum number of results to return
  public var numResults: Int

  /// Minimum similarity threshold (0.0-1.0)
  public var threshold: Float?

  public init(
    numResults: Int = 10,
    threshold: Float? = nil
  ) throws {
    // Validate numResults
    guard numResults > 0 else {
      throw VecturaError.invalidInput("numResults must be greater than 0, got \(numResults)")
    }

    // Validate threshold range if provided
    if let threshold = threshold {
      guard threshold >= 0.0 && threshold <= 1.0 else {
        throw VecturaError.invalidInput("threshold must be between 0.0 and 1.0, got \(threshold)")
      }
    }

    self.numResults = numResults
    self.threshold = threshold
  }
}
