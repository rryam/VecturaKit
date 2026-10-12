import Foundation
import VecturaKit

/// EmbeddingGemma 2 text indexing and text/image retrieval using a caller-owned llama.cpp OpenAI-compatible server.
/// The server owns GGUF tokenization, special tokens, mean pooling and the learned projection.
/// `embed(text:)` and `embed(texts:)` embed documents; `embedQuery(text:)` embeds search queries.
public actor EmbeddingGemma2Embedder: VecturaEmbedder {
  public nonisolated let configuration: EmbeddingGemma2Configuration
  private let backend: OpenAICompatibleEmbedder
  private let modelsURL: URL
  private let session: URLSession
  private let timeoutInterval: TimeInterval

  public init(
    configuration: EmbeddingGemma2Configuration,
    baseURL: String = "http://127.0.0.1:8080/v1",
    timeoutInterval: TimeInterval = 120,
    session: URLSession = .shared
  ) throws {
    guard let url = URL(string: baseURL),
      ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
      url.host != nil, url.query == nil, url.fragment == nil,
      timeoutInterval.isFinite, timeoutInterval > 0
    else {
      throw OpenAICompatibleEmbedderError.invalidInput("Invalid server URL or timeout")
    }
    self.configuration = configuration
    self.modelsURL = url.appendingPathComponent("models")
    self.session = session
    self.timeoutInterval = timeoutInterval
    self.backend = OpenAICompatibleEmbedder(
      baseURL: baseURL,
      model: configuration.modelAlias,
      timeoutInterval: timeoutInterval,
      retryAttempts: 0,
      retryBaseDelaySeconds: 1,
      session: session
    )
  }

  public var dimension: Int {
    get async throws { configuration.dimensions.rawValue }
  }

  public nonisolated var embeddingSpaceIdentifier: String? { configuration.embeddingSpaceIdentifier }

  public func embed(texts: [String]) async throws -> [[Float]] {
    try validate(texts)
    return try await embedFormatted(texts.map { "title: none | text: " + $0 })
  }

  public func embedQuery(text: String) async throws -> [Float] {
    try validate([text])
    let vectors = try await embedFormatted(["task: search result | query: " + text])
    return vectors[0]
  }

  /// Image formats supported by the image-query adapter.
  public enum ImageFormat: String, Sendable {
    case jpeg = "image/jpeg"
    case png = "image/png"
  }

  /// Embeds image bytes for `database.search(query: .vector(imageVector))` against stored text.
  /// Images are not saved or indexed. Input is limited to 10 MiB; the server validates image decoding.
  /// Requires `enableImageQueries` and a server advertising image input with the matching projector.
  public func embedImageQuery(data: Data, format: ImageFormat) async throws -> [Float] {
    try Task.checkCancellation()
    guard configuration.enableImageQueries else {
      throw OpenAICompatibleEmbedderError.invalidInput("Image queries must be explicitly enabled")
    }
    guard !data.isEmpty, data.count <= 10 * 1024 * 1024 else {
      throw OpenAICompatibleEmbedderError.invalidInput("Image data must be nonempty JPEG or PNG bytes, at most 10 MiB")
    }
    let signature: [UInt8] = format == .png ? [137, 80, 78, 71, 13, 10, 26, 10] : [255, 216, 255]
    guard data.starts(with: signature) else {
      throw OpenAICompatibleEmbedderError.invalidInput("Image bytes do not match the specified format")
    }
    try await validateServerModel(requiringImage: true)
    let body: [String: Any] = [
      "model": configuration.modelAlias,
      "input": [["content": [[
        "type": "image_url",
        "image_url": ["url": "data:\(format.rawValue);base64," + data.base64EncodedString()]
      ]]]]
    ]
    let vectors = try await backend.embed(
      requestBody: JSONSerialization.data(withJSONObject: body), expectedCount: 1, requireValidIndices: true
    )
    try Task.checkCancellation()
    return try normalize(vectors[0])
  }

  private func validate(_ texts: [String]) throws {
    try Task.checkCancellation()
    guard !texts.isEmpty, texts.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
      throw OpenAICompatibleEmbedderError.invalidInput("Inputs must contain nonempty text")
    }
  }

  private func embedFormatted(_ texts: [String]) async throws -> [[Float]] {
    var result: [[Float]] = []
    for offset in stride(from: 0, to: texts.count, by: configuration.batchSize) {
      try Task.checkCancellation()
      try await validateServerModel()
      let end = min(offset + configuration.batchSize, texts.count)
      let vectors = try await backend.embed(texts: Array(texts[offset..<end]), requireValidIndices: true)
      try Task.checkCancellation()
      result.append(contentsOf: try vectors.map(normalize))
    }
    return result
  }

  private func validateServerModel(requiringImage: Bool = false) async throws {
    var request = URLRequest(url: modelsURL)
    request.timeoutInterval = timeoutInterval
    let (data, response) = try await session.data(for: request)
    guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
      throw OpenAICompatibleEmbedderError.httpError(
        statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1,
        message: "Unable to verify llama-server model"
      )
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let models = try decoder.decode(ServerModelList.self, from: data)
    guard let model = models.data.first(where: { $0.id == configuration.modelAlias }) else {
      throw OpenAICompatibleEmbedderError.invalidInput("Server must load the selected GGUF with the configured alias")
    }
    if requiringImage, model.architecture?.inputModalities.contains("image") != true {
      throw OpenAICompatibleEmbedderError.invalidInput("Server must advertise image input; load the matching projector")
    }
  }

  private struct ServerModelList: Decodable { let data: [ServerModel] }
  private struct ServerModel: Decodable {
    let id: String
    let architecture: ServerArchitecture?
  }
  private struct ServerArchitecture: Decodable {
    let inputModalities: [String]
  }

  private func normalize(_ vector: [Float]) throws -> [Float] {
    guard vector.count == 768 else {
      throw VecturaError.dimensionMismatch(expected: 768, got: vector.count)
    }
    guard vector.allSatisfy(\.isFinite) else {
      throw OpenAICompatibleEmbedderError.invalidInput("Server returned nonfinite embedding values")
    }
    let truncated = Array(vector.prefix(configuration.dimensions.rawValue))
    let norm = sqrt(truncated.reduce(0.0) { $0 + Double($1) * Double($1) })
    guard norm > 0, norm.isFinite else {
      throw OpenAICompatibleEmbedderError.invalidInput("Server returned a zero or invalid embedding")
    }
    return truncated.map { Float(Double($0) / norm) }
  }
}
