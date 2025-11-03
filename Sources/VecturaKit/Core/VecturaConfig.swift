import Foundation

/// Configuration options for Vectura vector database.
public struct VecturaConfig: Sendable {
  
  /// Options for similarity search.
  public struct SearchOptions: Sendable {
    
    /// The default number of results to return.
    public var defaultNumResults: Int = 10
    
    /// The minimum similarity threshold.
    public var minThreshold: Float?
    
    private var _hybridWeight: Float = 0.5
    
    /// Weight for vector similarity in hybrid search (0.0-1.0)
    /// BM25 weight will be (1-hybridWeight)
    /// Values outside the range will be clamped to [0.0, 1.0]
    public var hybridWeight: Float {
      get { _hybridWeight }
      set { _hybridWeight = max(0.0, min(1.0, newValue)) }
    }
    
    /// BM25 parameters
    public var k1: Float = 1.2
    public var b: Float = 0.75
    
    /// BM25 score normalization factor. BM25 scores are divided by this value
    /// to normalize them to a 0-1 range for hybrid search. Adjust based on
    /// your corpus size and typical BM25 score ranges.
    public var bm25NormalizationFactor: Float = 10.0
    
    public init(
      defaultNumResults: Int = 10,
      minThreshold: Float? = nil,
      hybridWeight: Float = 0.5,
      k1: Float = 1.2,
      b: Float = 0.75,
      bm25NormalizationFactor: Float = 10.0
    ) {
      self.defaultNumResults = defaultNumResults
      self.minThreshold = minThreshold
      self.hybridWeight = hybridWeight
      self.k1 = k1
      self.b = b
      self.bm25NormalizationFactor = bm25NormalizationFactor
    }
  }
  
  /// Memory management strategy for handling documents in VecturaKit.
  ///
  /// This enum defines how VecturaKit loads and manages documents in memory,
  /// allowing you to optimize performance based on your dataset size.
  public enum MemoryStrategy: Equatable, Sendable {
    /// Default threshold for automatic strategy switching.
    public static let defaultAutomaticThreshold = 10_000
    
    /// Default candidate multiplier for indexed mode.
    /// This value balances between search accuracy and performance.
    /// - Higher values (15-20): Better recall at the cost of slower searches
    /// - Lower values (5-10): Faster searches but may miss some relevant results
    public static let defaultCandidateMultiplier = 10
    
    /// Default batch size for concurrent document loading in indexed mode.
    /// Documents are loaded in batches to balance concurrency and memory usage.
    /// - Smaller batches (50-100): More concurrent tasks, higher overhead
    /// - Larger batches (100-200): Fewer concurrent tasks, better throughput
    public static let defaultBatchSize = 100
    
    /// Default maximum number of concurrent batch loading operations.
    /// Limits the number of simultaneous storage queries to prevent resource exhaustion.
    public static let defaultMaxConcurrentBatches = 4
    
    /// Automatic mode: Selects the optimal strategy based on document count.
    ///
    /// - Uses `fullMemory` for datasets < threshold
    /// - Uses `indexed` mode for larger datasets with specified parameters
    ///
    /// - Parameters:
    ///   - threshold: Document count threshold for switching strategies (default: 10,000)
    ///   - candidateMultiplier: Candidate pool size for indexed mode (default: 10)
    ///   - batchSize: Batch size for concurrent loading in indexed mode (default: 100)
    ///   - maxConcurrentBatches: Max concurrent batches in indexed mode (default: 4)
    ///
    /// This is the recommended default for most use cases. When the dataset size
    /// exceeds the threshold and switches to indexed mode, it will use the
    /// specified parameters for indexed search operations.
    case automatic(
      threshold: Int = defaultAutomaticThreshold,
      candidateMultiplier: Int = defaultCandidateMultiplier,
      batchSize: Int = defaultBatchSize,
      maxConcurrentBatches: Int = defaultMaxConcurrentBatches
    )
    
    /// Full memory mode: Load all documents into memory.
    ///
    /// Best for small to medium datasets (< 100,000 documents) where:
    /// - Fast search performance is critical (< 10ms)
    /// - Memory usage is not a constraint
    /// - Dataset fits comfortably in RAM
    ///
    /// Memory usage: ~180-200 KB per document (with 384-dimensional embeddings)
    case fullMemory
    
    /// Indexed mode: Use storage-layer indexing with on-demand loading.
    ///
    /// Best for large datasets (> 100,000 documents) where:
    /// - Memory efficiency is important
    /// - Moderate search latency is acceptable
    /// - Storage provider supports `IndexedVecturaStorage`
    ///
    /// - Parameters:
    ///   - candidateMultiplier: Candidate pool size = topK × multiplier.
    ///     Higher values improve accuracy but increase search time.
    ///     Recommended: 5-20 (default: 10)
    ///   - batchSize: Number of documents to load per batch during concurrent loading.
    ///     Smaller batches increase concurrency but may have overhead.
    ///     Recommended: 50-200 (default: 100)
    ///   - maxConcurrentBatches: Maximum number of concurrent batch loading operations.
    ///     Prevents resource exhaustion by limiting simultaneous storage queries.
    ///     Recommended: 2-8 (default: 4)
    ///
    /// - Note: If the storage provider doesn't implement `IndexedVecturaStorage`,
    ///   VecturaKit will automatically fall back to `fullMemory` mode.
    ///
    /// - Important: Parameter validation occurs when initializing VecturaKit, not during
    ///   config creation. All parameters must be positive integers.
    case indexed(
      candidateMultiplier: Int = defaultCandidateMultiplier,
      batchSize: Int = defaultBatchSize,
      maxConcurrentBatches: Int = defaultMaxConcurrentBatches
    )
  }
  
  /// The name of the database instance.
  public let name: String
  
  /// A custom directory where the database should be stored.
  /// Will be created if it doesn't exist, database contents are stored in a subdirectory named after ``name``.
  public let directoryURL: URL?
  
  /// The dimension of vectors to be stored. If nil, will be auto-detected from the model.
  public let dimension: Int?
  
  /// Memory management strategy for handling large-scale datasets.
  ///
  /// This setting controls how VecturaKit loads and manages documents in memory.
  /// Choose a strategy based on your dataset size and performance requirements.
  ///
  /// - Note: The strategy is fixed at initialization time and cannot be changed after
  ///   the VecturaKit instance is created.
  public let memoryStrategy: MemoryStrategy

  /// Search configuration options.
  public var searchOptions: SearchOptions
  
  public init(
    name: String,
    directoryURL: URL? = nil,
    dimension: Int? = nil,
    searchOptions: SearchOptions = .init(),
    memoryStrategy: MemoryStrategy = .automatic()
  ) {
    self.name = name
    self.directoryURL = directoryURL
    self.dimension = dimension
    self.searchOptions = searchOptions
    self.memoryStrategy = memoryStrategy
  }
}
