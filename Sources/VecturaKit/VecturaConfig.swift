import Foundation

/// Configuration options for Vectura vector database.
public struct VecturaConfig {

  /// The name of the database instance.
  public let name: String

  /// A custom directory where the database should be stored.
  /// Will be created if it doesn't exist, database contents are stored in a subdirectory named after ``name``.
  public let directoryURL: URL?

  /// The dimension of vectors to be stored.
  public let dimension: Int

  /// Options for similarity search.
  public struct SearchOptions {
    /// The default number of results to return.
    public var defaultNumResults: Int = 10

    /// The minimum similarity threshold.
    public var minThreshold: Float?

    /// Weight for vector similarity in hybrid search (0.0-1.0)
    /// BM25 weight will be (1-hybridWeight)
    public var hybridWeight: Float = 0.5

    /// BM25 parameters
    public var k1: Float = 1.2
    public var b: Float = 0.75

    public init(
      defaultNumResults: Int = 10,
      minThreshold: Float? = nil,
      hybridWeight: Float = 0.5,
      k1: Float = 1.2,
      b: Float = 0.75
    ) {
      self.defaultNumResults = defaultNumResults
      self.minThreshold = minThreshold
      self.hybridWeight = hybridWeight
      self.k1 = k1
      self.b = b
    }
  }

  /// Search configuration options.
  public var searchOptions: SearchOptions

  public init(
    name: String,
    directoryURL: URL? = nil,
    dimension: Int,
    searchOptions: SearchOptions = SearchOptions()
  ) {
    self.name = name
    self.directoryURL = directoryURL
    self.dimension = dimension
    self.searchOptions = searchOptions
  }
}
