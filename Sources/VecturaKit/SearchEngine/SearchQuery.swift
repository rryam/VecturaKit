import Foundation

/// Search query types supported by search engines
public enum SearchQuery: Sendable {

  /// Text-based query
  case text(String)

  /// Vector-based query
  case vector([Float])
}

// MARK: - ExpressibleByStringLiteral

extension SearchQuery: ExpressibleByStringLiteral {
  /// Create a text query from a string literal
  ///
  /// This allows writing:
  /// ```swift
  /// vectura.search(query: "hello world")
  /// ```
  /// instead of:
  /// ```swift
  /// vectura.search(query: .text("hello world"))
  /// ```
  public init(stringLiteral value: String) {
    self = .text(value)
  }
}

// MARK: - ExpressibleByArrayLiteral

extension SearchQuery: ExpressibleByArrayLiteral {
  /// Create a vector query from an array literal
  ///
  /// This allows writing:
  /// ```swift
  /// vectura.search(query: [0.1, 0.2, 0.3])
  /// ```
  /// instead of:
  /// ```swift
  /// vectura.search(query: .vector([0.1, 0.2, 0.3]))
  /// ```
  ///
  /// - Note: This only works with array literals, not with variables.
  ///   For variables, you still need to use `.vector(embedding)`.
  public init(arrayLiteral elements: Float...) {
    self = .vector(elements)
  }
}
