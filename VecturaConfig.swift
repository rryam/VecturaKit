import Foundation

/// Configuration options for Vectura vector database.
public struct VecturaConfig {
    /// The name of the database instance.
    public let name: String
    
    /// The dimension of vectors to be stored.
    public let dimension: Int
    
    /// Options for similarity search.
    public struct SearchOptions {
        /// The default number of results to return.
        public var defaultNumResults: Int = 10
        
        /// The minimum similarity threshold.
        public var minThreshold: Float?
        
        public init(defaultNumResults: Int = 10, minThreshold: Float? = nil) {
            self.defaultNumResults = defaultNumResults
            self.minThreshold = minThreshold
        }
    }
    
    /// Search configuration options.
    public var searchOptions: SearchOptions
    
    public init(
        name: String,
        dimension: Int,
        searchOptions: SearchOptions = SearchOptions()
    ) {
        self.name = name
        self.dimension = dimension
        self.searchOptions = searchOptions
    }
}

// End of file. No additional code.
