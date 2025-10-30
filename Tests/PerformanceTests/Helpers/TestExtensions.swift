import Foundation

// MARK: - String Extensions

extension String {
    /// Repeats a string multiple times.
    ///
    /// Example:
    /// ```swift
    /// "=" * 5  // "====="
    /// ```
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
