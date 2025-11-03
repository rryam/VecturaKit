//
//  ArrayExtensions.swift
//  VecturaKit
//
//  Created by Claude Code on 2025-11-03.
//

import Foundation

// MARK: - Array Extension for Chunking

extension Array {
    /// Splits the array into chunks of the specified size.
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
