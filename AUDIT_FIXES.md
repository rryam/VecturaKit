# VecturaKit Audit Fixes Checklist

## High Priority

### 1. Missing input validation for MemoryStrategy parameters
- [ ] Add validation for `candidateMultiplier > 0` in VecturaConfig.init
- [ ] Add validation for `batchSize > 0` in VecturaConfig.init
- [ ] Add validation for `maxConcurrentBatches > 0` in VecturaConfig.init
- [ ] Add validation for `threshold` parameter in automatic strategy
- [ ] Add tests for invalid configuration parameters
- [ ] Update VecturaError to include configuration validation errors

**Files affected:**
- `Sources/VecturaKit/Core/VecturaConfig.swift`
- `Sources/VecturaKit/Core/VecturaError.swift`
- `Tests/VecturaKitTests/VecturaKitTests.swift`

**Implementation details:**
```swift
// In VecturaConfig.init, add validation for memory strategy parameters
switch memoryStrategy {
case .indexed(let multiplier, let batch, let maxConcurrent),
     .automatic(_, let multiplier, let batch, let maxConcurrent):
  guard multiplier > 0 else {
    throw VecturaError.invalidInput("candidateMultiplier must be positive, got \(multiplier)")
  }
  guard batch > 0 else {
    throw VecturaError.invalidInput("batchSize must be positive, got \(batch)")
  }
  guard maxConcurrent > 0 else {
    throw VecturaError.invalidInput("maxConcurrentBatches must be positive, got \(maxConcurrent)")
  }
case .fullMemory:
  break
}
```

---

## Medium Priority

### 2. File permission gaps
- [ ] Set explicit directory permissions (0o700) when creating storage directories
- [ ] Verify file permissions were successfully set after setting them
- [ ] Add cross-platform permission verification
- [ ] Add error handling for permission setting failures
- [ ] Add tests for file permission verification

**Files affected:**
- `Sources/VecturaKit/Core/VecturaKit.swift` (createStorageDirectory)
- `Sources/VecturaKit/Storage/FileStorageProvider.swift` (saveDocument)

**Implementation details:**
```swift
// In createStorageDirectory
try FileManager.default.createDirectory(
  at: storageDirectory,
  withIntermediateDirectories: true,
  attributes: [.posixPermissions: 0o700]  // Owner only
)

// In saveDocument, verify permissions after setting
#if !os(iOS) && !os(tvOS) && !os(watchOS) && !os(visionOS)
let attrs = try FileManager.default.attributesOfItem(atPath: documentURL.path(percentEncoded: false))
guard let perms = attrs[.posixPermissions] as? NSNumber,
      perms.uint16Value == 0o600 else {
  throw VecturaError.saveFailed("Failed to set secure file permissions")
}
#endif
```

### 3. String-based model type detection
- [ ] Add explicit ModelType enum to VecturaModelSource
- [ ] Update isModel2VecModel to use optional explicit type first
- [ ] Fall back to string matching only if type not specified
- [ ] Add documentation about specifying model type explicitly
- [ ] Update tests to use explicit model types

**Files affected:**
- `Sources/VecturaKit/Embedder/VecturaModelSource.swift`
- `Sources/VecturaKit/Embedder/SwiftEmbedder.swift`
- `Tests/VecturaKitTests/VecturaKitTests.swift`

**Implementation details:**
```swift
// In VecturaModelSource
public enum VecturaModelSource: Sendable {
  case id(_ id: String, type: ModelType? = nil)
  case folder(_ url: URL, type: ModelType? = nil)

  public enum ModelType {
    case bert
    case model2vec
  }
}

// In SwiftEmbedder
private func isModel2VecModel(_ source: VecturaModelSource) -> Bool {
  // Check explicit type first
  switch source {
  case .id(_, let type), .folder(_, let type):
    if let type = type {
      return type == .model2vec
    }
  }

  // Fall back to string matching
  let modelId = source.description
  return modelId.contains("minishlab") ||
         modelId.contains("potion") ||
         modelId.contains("model2vec") ||
         modelId.contains("M2V")
}
```

### 4. Silent document loading failures
- [ ] Create LoadResult struct with documents and failures arrays
- [ ] Update FileStorageProvider.loadDocuments to return LoadResult
- [ ] Add logging for failed document loads with file paths
- [ ] Update callers to handle LoadResult appropriately
- [ ] Add option to fail-fast vs. partial loading
- [ ] Add tests for partial loading scenarios

**Files affected:**
- `Sources/VecturaKit/Storage/FileStorageProvider.swift`
- `Sources/VecturaKit/Storage/VecturaStorage.swift` (protocol)
- `Sources/VecturaKit/Core/VecturaKit.swift`
- `Sources/VecturaKit/SearchEngine/VectorSearchEngine.swift`

**Implementation details:**
```swift
// New struct
public struct DocumentLoadResult {
  public let documents: [VecturaDocument]
  public let failures: [(path: String, error: Error)]

  public var hasFailures: Bool {
    !failures.isEmpty
  }
}

// Update protocol to allow both approaches
extension VecturaStorage {
  // Existing method throws if any failures
  func loadDocuments() async throws -> [VecturaDocument]

  // New method returns partial results
  func loadDocumentsWithFailures() async throws -> DocumentLoadResult
}
```

---

## Low Priority

### 5. Unsafe array indexing in addDocument
- [ ] Replace `ids[0]` with safe `ids.first` + guard
- [ ] Add proper error handling for empty results
- [ ] Replace `shape[1]` with safe array access in SwiftEmbedder
- [ ] Add tests for edge cases

**Files affected:**
- `Sources/VecturaKit/Core/VecturaKit.swift`
- `Sources/VecturaKit/Embedder/SwiftEmbedder.swift`

**Implementation details:**
```swift
// In VecturaKit.addDocument
public func addDocument(text: String, id: UUID? = nil) async throws -> UUID {
  let ids = try await addDocuments(texts: [text], ids: id.map { [$0] })
  guard let firstId = ids.first else {
    throw VecturaError.invalidInput("Failed to add document: no ID returned")
  }
  return firstId
}

// In SwiftEmbedder
guard shape.count == 2, let dimension = shape.last else {
  throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
}
```

### 6. Code duplication - normalizeEmbedding and l2Norm
- [ ] Create shared VectorMath utility enum
- [ ] Move normalizeEmbedding to VectorMath
- [ ] Move l2Norm to VectorMath
- [ ] Update VecturaKit to use VectorMath
- [ ] Update VectorSearchEngine to use VectorMath
- [ ] Add tests for VectorMath utilities

**Files affected:**
- `Sources/VecturaKit/Utilities/VectorMath.swift` (new file)
- `Sources/VecturaKit/Core/VecturaKit.swift`
- `Sources/VecturaKit/SearchEngine/VectorSearchEngine.swift`

**Implementation details:**
```swift
// New file: Sources/VecturaKit/Utilities/VectorMath.swift
import Accelerate
import Foundation

public enum VectorMath {
  /// Normalizes an embedding vector to unit length (L2 normalization)
  public static func normalizeEmbedding(_ embedding: [Float]) throws -> [Float] {
    let norm = l2Norm(embedding)

    guard norm > 1e-10 else {
      throw VecturaError.invalidInput("Cannot normalize zero-norm embedding vector")
    }

    var divisor = norm
    var normalized = [Float](repeating: 0, count: embedding.count)
    vDSP_vsdiv(embedding, 1, &divisor, &normalized, 1, vDSP_Length(embedding.count))
    return normalized
  }

  /// Computes the L2 norm of a vector
  public static func l2Norm(_ v: [Float]) -> Float {
    var sumSquares: Float = 0
    vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
    return sqrt(sumSquares)
  }
}
```

### 7. Missing concurrent access tests
- [ ] Add test for concurrent document addition (100 parallel adds)
- [ ] Add test for concurrent reads while writing
- [ ] Add test for concurrent updates to same document
- [ ] Add test for concurrent search operations
- [ ] Add stress test with mixed operations
- [ ] Add test for BM25Index actor isolation

**Files affected:**
- `Tests/VecturaKitTests/ConcurrencyTests.swift` (new file)

**Implementation details:**
```swift
// New test file for concurrency
@Test("Concurrent document addition")
func concurrentAddDocuments() async throws {
  let vectura = try await VecturaKit(...)

  await withTaskGroup(of: Result<UUID, Error>.self) { group in
    for i in 0..<100 {
      group.addTask {
        do {
          let id = try await vectura.addDocument(text: "Doc \(i)")
          return .success(id)
        } catch {
          return .failure(error)
        }
      }
    }

    var successCount = 0
    for await result in group {
      if case .success = result {
        successCount += 1
      }
    }

    #expect(successCount == 100)
  }

  #expect(try await vectura.documentCount == 100)
}
```

---

## Progress Tracking

**High Priority:** 0/1 completed
**Medium Priority:** 0/4 completed
**Low Priority:** 0/3 completed

**Total:** 0/8 completed (0%)
