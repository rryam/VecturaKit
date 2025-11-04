import Foundation

/// Protocol for storage implementations that support in-memory caching.
///
/// This protocol extends `VecturaStorage` with caching capabilities. Unlike
/// providing default implementations, storage providers are expected to implement
/// the caching logic themselves to ensure thread safety (especially for actors).
///
/// ## Usage
///
/// ```swift
/// public actor MyStorage: CachableVecturaStorage {
///   private var cache: [UUID: VecturaDocument] = [:]
///   private let cacheEnabled: Bool
///
///   public func loadDocuments() async throws -> [VecturaDocument] {
///     if cacheEnabled && !cache.isEmpty {
///       return Array(cache.values)
///     }
///     let documents = try await loadDocumentsFromStorage()
///     if cacheEnabled {
///       cache = Dictionary(uniqueKeysWithValues: documents.map { ($0.id, $0) })
///     }
///     return documents
///   }
///   // ... other methods
/// }
/// ```
public protocol CachableVecturaStorage: VecturaStorage {

  /// Loads documents from the underlying storage without using the cache
  func loadDocumentsFromStorage() async throws -> [VecturaDocument]

  /// Saves a document to the underlying storage (implementations should update cache)
  func saveDocumentToStorage(_ document: VecturaDocument) async throws

  /// Deletes a document from the underlying storage (implementations should update cache)
  func deleteDocumentFromStorage(withID id: UUID) async throws
}
