import Foundation
import Testing
@testable import VecturaKit

/// Tests for concurrent access patterns and thread safety
@Suite("Concurrency Tests")
struct ConcurrencyTests {

  private func makeTestDirectory() throws -> (URL, () -> Void) {
    let directory = URL(filePath: NSTemporaryDirectory())
      .appendingPathComponent("ConcurrencyTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o755]
    )
    let cleanup = {
      _ = try? FileManager.default.removeItem(at: directory)
    }
    return (directory, cleanup)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
  private func makeEmbedder() -> SwiftEmbedder {
    SwiftEmbedder(modelSource: .default)
  }

  /// Test concurrent document additions to ensure thread safety
  @Test("Concurrent document addition")
  func concurrentAddDocuments() async throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-concurrent-add",
      directoryURL: directory,
      searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 10)
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    // Add 20 documents concurrently
    await withTaskGroup(of: Result<UUID, Error>.self) { group in
      for i in 0..<20 {
        group.addTask {
          do {
            let id = try await vectura.addDocument(text: "Concurrent document \(i)")
            return .success(id)
          } catch {
            return .failure(error)
          }
        }
      }

      var successCount = 0
      var failureCount = 0

      for await result in group {
        switch result {
        case .success:
          successCount += 1
        case .failure:
          failureCount += 1
        }
      }

      #expect(successCount == 20)
      #expect(failureCount == 0)
    }

    let count = try await vectura.documentCount
    #expect(count == 20)
  }

  /// Test concurrent reads while writing
  @Test("Concurrent reads during writes")
  func concurrentReadsAndWrites() async throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-concurrent-read-write",
      directoryURL: directory,
      searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 5)
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    // Pre-populate with some documents
    _ = try await vectura.addDocuments(texts: [
      "Initial document 1",
      "Initial document 2",
      "Initial document 3"
    ])

    // Concurrently read and write
    await withTaskGroup(of: Result<Void, Error>.self) { group in
      // 10 write tasks
      for i in 0..<10 {
        group.addTask {
          do {
            _ = try await vectura.addDocument(text: "New document \(i)")
            return .success(())
          } catch {
            return .failure(error)
          }
        }
      }

      // 10 read tasks
      for i in 0..<10 {
        group.addTask {
          do {
            _ = try await vectura.search(
              query: SearchQuery.text("document \(i)"),
              numResults: 5
            )
            return .success(())
          } catch {
            return .failure(error)
          }
        }
      }

      var errorCount = 0
      for await result in group {
        if case .failure = result {
          errorCount += 1
        }
      }

      #expect(errorCount == 0)
    }

    let finalCount = try await vectura.documentCount
    #expect(finalCount == 13)  // 3 initial + 10 new
  }

  /// Test concurrent search operations
  @Test("Concurrent search operations")
  func concurrentSearches() async throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }

    let (directory, cleanup) = try makeTestDirectory()
    defer { cleanup() }

    let config = try VecturaConfig(
      name: "test-concurrent-searches",
      directoryURL: directory,
      searchOptions: VecturaConfig.SearchOptions(defaultNumResults: 5)
    )
    let vectura = try await VecturaKit(config: config, embedder: makeEmbedder())

    // Add some documents
    _ = try await vectura.addDocuments(texts: [
      "Machine learning algorithms",
      "Deep neural networks",
      "Natural language processing",
      "Computer vision systems",
      "Data science tools"
    ])

    // Perform 20 concurrent searches
    await withTaskGroup(of: Result<Int, Error>.self) { group in
      for i in 0..<20 {
        group.addTask {
          do {
            let results = try await vectura.search(
              query: SearchQuery.text("query \(i)"),
              numResults: 3
            )
            return .success(results.count)
          } catch {
            return .failure(error)
          }
        }
      }

      var errorCount = 0
      for await result in group {
        if case .failure = result {
          errorCount += 1
        }
      }

      #expect(errorCount == 0)
    }
  }
}
