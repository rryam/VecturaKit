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
  ) {
    self.numResults = numResults
    self.threshold = threshold
  }
}
