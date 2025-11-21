import Accelerate
import Foundation

/// Utility functions for vector mathematics operations.
public enum VectorMath {

  /// Normalizes an embedding vector to unit length (L2 normalization).
  ///
  /// - Parameter embedding: The vector to normalize
  /// - Returns: The normalized vector with unit length
  /// - Throws: `VecturaError.invalidInput` if the vector has zero norm
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

  /// Computes the L2 norm (Euclidean length) of a vector.
  ///
  /// - Parameter v: The input vector
  /// - Returns: The L2 norm of the vector
  public static func l2Norm(_ v: [Float]) -> Float {
    var sumSquares: Float = 0
    vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
    return sqrt(sumSquares)
  }
}
