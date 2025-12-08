import Foundation

// MARK: - Concurrent Collection Processing

extension Sequence where Element: Sendable {

  /// Concurrently maps elements with controlled parallelism using a sliding window pattern.
  ///
  /// This method processes elements concurrently while limiting the number of simultaneous
  /// operations to prevent resource exhaustion (file handles, memory, etc.).
  ///
  /// - Parameters:
  ///   - maxConcurrency: Maximum number of concurrent operations (default: 50)
  ///   - transform: Async throwing closure that transforms each element. Return `nil` to skip.
  /// - Returns: Array of non-nil transformed results
  /// - Throws: Rethrows any error from the transform closure
  ///
  /// ## Example
  ///
  /// ```swift
  /// let urls: [URL] = getFileURLs()
  /// let documents = try await urls.concurrentMap(maxConcurrency: 50) { url in
  ///   try? JSONDecoder().decode(Document.self, from: Data(contentsOf: url))
  /// }
  /// ```
  ///
  /// ## Performance Characteristics
  ///
  /// - Maintains exactly `maxConcurrency` tasks running at any time
  /// - Uses structured concurrency with no semaphores or locks
  /// - Memory efficient: processes results as they complete
  @inlinable
  public func concurrentMap<T: Sendable>(
    maxConcurrency: Int = 50,
    _ transform: @Sendable @escaping (Element) async throws -> T?
  ) async rethrows -> [T] {
    try await withThrowingTaskGroup(of: T?.self) { group in
      var results: [T] = []
      var iterator = makeIterator()

      // Seed initial batch of tasks up to maxConcurrency
      var activeCount = 0
      while activeCount < maxConcurrency, let element = iterator.next() {
        group.addTask { try await transform(element) }
        activeCount += 1
      }

      // As each task completes, add the next one (sliding window)
      while let result = try await group.next() {
        if let value = result {
          results.append(value)
        }

        // Add next task if elements remain
        if let element = iterator.next() {
          group.addTask { try await transform(element) }
        }
      }

      return results
    }
  }

  /// Concurrently maps elements with controlled parallelism (non-throwing version).
  ///
  /// - Parameters:
  ///   - maxConcurrency: Maximum number of concurrent operations (default: 50)
  ///   - transform: Async closure that transforms each element. Return `nil` to skip.
  /// - Returns: Array of non-nil transformed results
  @inlinable
  public func concurrentMap<T: Sendable>(
    maxConcurrency: Int = 50,
    _ transform: @Sendable @escaping (Element) async -> T?
  ) async -> [T] {
    await withTaskGroup(of: T?.self) { group in
      var results: [T] = []
      var iterator = makeIterator()

      // Seed initial batch
      var activeCount = 0
      while activeCount < maxConcurrency, let element = iterator.next() {
        group.addTask { await transform(element) }
        activeCount += 1
      }

      // Sliding window: add new task as each completes
      for await result in group {
        if let value = result {
          results.append(value)
        }

        if let element = iterator.next() {
          group.addTask { await transform(element) }
        }
      }

      return results
    }
  }

  /// Concurrently executes a side-effect closure on each element with controlled parallelism.
  ///
  /// Use this when you need to perform async operations for their side effects
  /// (e.g., saving files, network requests) without collecting results.
  ///
  /// - Parameters:
  ///   - maxConcurrency: Maximum number of concurrent operations (default: 50)
  ///   - body: Async throwing closure to execute for each element
  /// - Throws: Rethrows the first error encountered
  ///
  /// ## Example
  ///
  /// ```swift
  /// try await documents.concurrentForEach(maxConcurrency: 20) { doc in
  ///   try await storage.saveDocument(doc)
  /// }
  /// ```
  @inlinable
  public func concurrentForEach(
    maxConcurrency: Int = 50,
    _ body: @Sendable @escaping (Element) async throws -> Void
  ) async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
      var iterator = makeIterator()

      // Seed initial batch
      var activeCount = 0
      while activeCount < maxConcurrency, let element = iterator.next() {
        group.addTask { try await body(element) }
        activeCount += 1
      }

      // Sliding window
      while try await group.next() != nil {
        if let element = iterator.next() {
          group.addTask { try await body(element) }
        }
      }
    }
  }

  /// Concurrently maps elements while preserving the original order.
  ///
  /// Unlike `concurrentMap`, this method guarantees that output order matches input order.
  /// Slightly higher memory overhead due to tracking indices.
  ///
  /// - Parameters:
  ///   - maxConcurrency: Maximum number of concurrent operations (default: 50)
  ///   - transform: Async throwing closure that transforms each element
  /// - Returns: Array of transformed results in the same order as input
  /// - Throws: Rethrows any error from the transform closure
  ///
  /// ## Example
  ///
  /// ```swift
  /// let urls = [url1, url2, url3]
  /// let data = try await urls.orderedConcurrentMap(maxConcurrency: 10) { url in
  ///   try await fetchData(from: url)
  /// }
  /// // data[0] corresponds to url1, data[1] to url2, etc.
  /// ```
  @inlinable
  public func orderedConcurrentMap<T: Sendable>(
    maxConcurrency: Int = 50,
    _ transform: @Sendable @escaping (Element) async throws -> T
  ) async rethrows -> [T] {
    let indexed = Array(self.enumerated())

    let results = try await indexed.concurrentMap(maxConcurrency: maxConcurrency) { item -> (Int, T)? in
      let result = try await transform(item.element)
      return (item.offset, result)
    }

    // Sort by original index and extract values
    return results
      .sorted { $0.0 < $1.0 }
      .map(\.1)
  }
}

// MARK: - Collection Chunking

extension Collection {

  /// Splits the collection into chunks of the specified size.
  ///
  /// - Parameter size: Maximum size of each chunk
  /// - Returns: Array of array slices, each containing up to `size` elements
  ///
  /// ## Example
  ///
  /// ```swift
  /// let items = [1, 2, 3, 4, 5, 6, 7]
  /// let chunks = items.chunked(into: 3)
  /// // [[1, 2, 3], [4, 5, 6], [7]]
  /// ```
  @inlinable
  public func chunked(into size: Int) -> [[Element]] {
    guard size > 0 else { return [] }

    return stride(from: 0, to: count, by: size).map { startOffset in
      let start = index(startIndex, offsetBy: startOffset)
      let end = index(start, offsetBy: size, limitedBy: endIndex) ?? endIndex
      return Array(self[start..<end])
    }
  }
}
