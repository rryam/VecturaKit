import Foundation

/// Text-index GGUF configuration for EmbeddingGemma 2 through a separately managed llama-server.
/// This does not download weights, start a server, or quantize a model at runtime.
public struct EmbeddingGemma2Configuration: Sendable, Equatable {
  /// Published text-weight artifacts. BF16 is full precision, Q8_0 is quantized.
  public enum Precision: String, Sendable, CaseIterable {
    case q8Zero = "Q8_0"
    case bf16 = "BF16"

    public var filename: String { "embeddinggemma-2-\(rawValue).gguf" }

    /// Exact artifact bytes at `modelRevision`; excludes runtime and activation memory.
    public var artifactSizeBytes: Int {
      switch self {
      case .q8Zero: 309_855_456
      case .bf16: 557_950_176
      }
    }

    public var sha256: String {
      switch self {
      case .q8Zero: "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
      case .bf16: "68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216"
      }
    }

    public var projectorFilename: String { "mmproj-" + filename }

    /// Additional published projector bytes for image queries (contains vision and audio encoders).
    public var projectorSizeBytes: Int {
      switch self {
      case .q8Zero: 554_821_024
      case .bf16: 982_074_784
      }
    }

    public var projectorSHA256: String {
      switch self {
      case .q8Zero: "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f"
      case .bf16: "d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766"
      }
    }

    public var projectorDownloadURL: URL {
      downloadURL.deletingLastPathComponent().appendingPathComponent(projectorFilename)
    }

    /// Pinned public upstream artifact. Verify the checksum before starting the server.
    public var downloadURL: URL {
      URL(string: "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/"
        + EmbeddingGemma2Configuration.modelRevision + "/" + filename)!
    }
  }

  /// Supported Matryoshka output sizes. Truncation is followed by client-side L2 normalization.
  public enum Dimensions: Int, Sendable, CaseIterable {
    case d128 = 128
    case d256 = 256
    case d512 = 512
    case d768 = 768
  }

  public static let modelRevision = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"
  public let precision: Precision
  public let dimensions: Dimensions
  public let batchSize: Int
  /// Enables image query embedding; the caller must load the matching projector.
  public let enableImageQueries: Bool

  public init(
    precision: Precision = .q8Zero,
    dimensions: Dimensions = .d768,
    batchSize: Int = 8,
    enableImageQueries: Bool = false
  ) throws {
    guard (1...32).contains(batchSize) else {
      throw OpenAICompatibleEmbedderError.invalidInput("Batch size must be between 1 and 32")
    }
    self.precision = precision
    self.dimensions = dimensions
    self.batchSize = batchSize
    self.enableImageQueries = enableImageQueries
  }

  /// Pass this alias to llama-server; the provider checks `/v1/models` before every embedding batch.
  public var modelAlias: String { "embeddinggemma-2-\(precision.rawValue)-\(Self.modelRevision)" }

  public var embeddingSpaceIdentifier: String {
    "vectura:embeddinggemma2:\(Self.modelRevision):\(precision.rawValue):\(dimensions.rawValue):search-v1:l2"
  }

  /// Arguments for a caller-owned llama-server (b11474 or later with EmbeddingGemma 2 support).
  /// Arguments are separate strings; pass directly to Process.arguments, without a shell.
  /// Configure a larger context/batch only after checking memory headroom.
  public func llamaServerArguments(modelDirectory: URL, port: UInt16 = 8080) -> [String] {
    var arguments = [
      "--model", modelDirectory.appendingPathComponent(precision.filename).path,
      "--alias", modelAlias,
      "--embedding", "--pooling", "mean",
      "--ctx-size", "512", "--batch-size", "512", "--ubatch-size", "512",
      "--parallel", "1", "--threads", "4", "--n-gpu-layers", "99",
      "--host", "127.0.0.1", "--port", String(port), "--no-webui"
    ]
    if enableImageQueries {
      arguments += ["--mmproj", modelDirectory.appendingPathComponent(precision.projectorFilename).path]
    }
    return arguments
  }
}
