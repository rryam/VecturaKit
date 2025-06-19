import Foundation
import Accelerate

/// Input validation utilities for VecturaKit
public enum VecturaValidation {
    /// Maximum allowed text length for documents to prevent memory issues
    static let maxTextLength = 1_000_000 // 1MB of text
    
    /// Maximum allowed database name length
    static let maxDatabaseNameLength = 255
    
    /// Validates a database name for security and filesystem compatibility
    /// - Parameter name: The database name to validate
    /// - Throws: VecturaError.invalidInput if the name is invalid
    public static func validateDatabaseName(_ name: String) throws {
        guard !name.isEmpty else {
            throw VecturaError.invalidInput("Database name cannot be empty")
        }
        
        guard name.count <= maxDatabaseNameLength else {
            throw VecturaError.invalidInput("Database name too long (max \(maxDatabaseNameLength) characters)")
        }
        
        // Check for path traversal attempts and invalid characters
        let invalidCharacters = CharacterSet(charactersIn: "../\\:*?\"<>|")
        guard name.rangeOfCharacter(from: invalidCharacters) == nil else {
            throw VecturaError.invalidInput("Database name contains invalid characters")
        }
        
        // Check for reserved names on different platforms
        let reservedNames = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
        guard !reservedNames.contains(name.uppercased()) else {
            throw VecturaError.invalidInput("Database name is reserved")
        }
    }
    
    /// Validates text input for document content
    /// - Parameter text: The text to validate
    /// - Throws: VecturaError.invalidInput if the text is invalid
    public static func validateDocumentText(_ text: String) throws {
        guard text.count <= maxTextLength else {
            throw VecturaError.invalidInput("Document text too long (max \(maxTextLength) characters)")
        }
    }
    
    /// Validates a collection of texts
    /// - Parameter texts: The texts to validate
    /// - Throws: VecturaError.invalidInput if any text is invalid
    public static func validateDocumentTexts(_ texts: [String]) throws {
        for text in texts {
            try validateDocumentText(text)
        }
    }
}

/// Vector mathematics utilities for VecturaKit
public enum VectorMath {
    /// Computes the L2 norm (Euclidean norm) of a vector
    /// - Parameter vector: The input vector
    /// - Returns: The L2 norm of the vector
    public static func l2Norm(_ vector: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        return sqrt(sumSquares)
    }
    
    /// Normalizes a vector using L2 normalization (unit vector)
    /// - Parameter vector: The input vector to normalize
    /// - Returns: The normalized vector
    public static func normalizeL2(_ vector: [Float]) -> [Float] {
        let norm = l2Norm(vector)
        guard norm > 0 else { return vector } // Avoid division by zero
        
        var divisor = norm + 1e-9 // Add small epsilon for numerical stability
        var normalized = [Float](repeating: 0, count: vector.count)
        vDSP_vsdiv(vector, 1, &divisor, &normalized, 1, vDSP_Length(vector.count))
        return normalized
    }
    
    /// Computes the dot product of two vectors
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: The dot product of the vectors
    public static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
}

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
  ) throws {
    // Validate database name for security
    try VecturaValidation.validateDatabaseName(name)
    
    // Validate dimension
    guard dimension > 0 else {
      throw VecturaError.invalidInput("Dimension must be positive")
    }
    
    guard dimension <= 100_000 else {
      throw VecturaError.invalidInput("Dimension too large (max 100,000)")
    }
    
    self.name = name
    self.directoryURL = directoryURL
    self.dimension = dimension
    self.searchOptions = searchOptions
  }
}
