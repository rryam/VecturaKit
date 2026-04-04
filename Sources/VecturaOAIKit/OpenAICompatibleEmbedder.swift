import Foundation
import OSLog
import VecturaKit

/// A VecturaKit embedder that calls an OpenAI-compatible `/v1/embeddings` endpoint.
public actor OpenAICompatibleEmbedder: VecturaEmbedder {
  private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.vectura.oai", category: "embeddings")
  private let baseURL: String
  private let model: String
  private let apiKey: String?
  private let timeoutInterval: TimeInterval
  private let retryAttempts: Int
  private let retryBaseDelaySeconds: TimeInterval
  private let session: URLSession
  private var cachedDimension: Int?

  public init(
    baseURL: String,
    model: String,
    apiKey: String? = nil,
    timeoutInterval: TimeInterval,
    retryAttempts: Int,
    retryBaseDelaySeconds: TimeInterval,
    session: URLSession = .shared
  ) {
    self.baseURL = baseURL
    self.model = model
    self.apiKey = apiKey
    self.timeoutInterval = timeoutInterval
    self.retryAttempts = retryAttempts
    self.retryBaseDelaySeconds = retryBaseDelaySeconds
    self.session = session
  }

  public var dimension: Int {
    get async throws {
      if let cachedDimension {
        return cachedDimension
      }

      let result = try await embed(texts: ["hello"])
      let dimension = result.first?.count ?? 0
      cachedDimension = dimension
      return dimension
    }
  }

  public func embed(texts: [String]) async throws -> [[Float]] {
    guard !texts.isEmpty else {
      throw OpenAICompatibleEmbedderError.invalidInput("Cannot embed empty array of texts")
    }

    for (index, text) in texts.enumerated() {
      guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw OpenAICompatibleEmbedderError.invalidInput(
          "Text at index \(index) cannot be empty or whitespace-only"
        )
      }
    }

    var request = URLRequest(url: try embeddingsURL())
    request.httpMethod = "POST"
    request.timeoutInterval = timeoutInterval
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey, !apiKey.isEmpty {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let body: [String: Any] = ["model": model, "input": texts]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, response) = try await OpenAICompatibleRateLimitRetry.data(
      for: request,
      operation: "embeddings request",
      logger: logger,
      retryAttempts: retryAttempts,
      retryBaseDelaySeconds: retryBaseDelaySeconds,
      session: session
    )

    guard let httpResponse = response as? HTTPURLResponse,
      200..<300 ~= httpResponse.statusCode
    else {
      let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
      let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
      throw OpenAICompatibleEmbedderError.httpError(statusCode: statusCode, message: errorMessage)
    }

    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let dataArray = json["data"] as? [[String: Any]]
    else {
      throw OpenAICompatibleEmbedderError.httpError(statusCode: 0, message: "Invalid response format")
    }

    let embeddings = dataArray
      .sorted { ($0["index"] as? Int ?? 0) < ($1["index"] as? Int ?? 0) }
      .compactMap { item -> [Float]? in
        guard let embedding = item["embedding"] as? [NSNumber] else {
          return nil
        }
        return embedding.map(\.floatValue)
      }

    guard embeddings.count == texts.count else {
      throw OpenAICompatibleEmbedderError.vectorCountMismatch(expected: texts.count, received: embeddings.count)
    }

    if let dimension = embeddings.first?.count {
      cachedDimension = dimension
    }

    return embeddings
  }

  private func embeddingsURL() throws -> URL {
    let trimmedBaseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
    guard
      !trimmedBaseURL.isEmpty,
      let rawURL = URL(string: trimmedBaseURL),
      rawURL.scheme != nil,
      var components = URLComponents(url: rawURL, resolvingAgainstBaseURL: false)
    else {
      throw OpenAICompatibleEmbedderError.invalidURL
    }

    let pathComponents = components.percentEncodedPath
      .split(separator: "/", omittingEmptySubsequences: true)

    if pathComponents.last == "embeddings" {
      guard let url = components.url else {
        throw OpenAICompatibleEmbedderError.invalidURL
      }
      return url
    }

    let updatedPath = "/" + (pathComponents + ["embeddings"]).joined(separator: "/")
    components.percentEncodedPath = updatedPath

    guard let url = components.url else {
      throw OpenAICompatibleEmbedderError.invalidURL
    }
    return url
  }
}
