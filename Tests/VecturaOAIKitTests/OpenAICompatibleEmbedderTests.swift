import Foundation
import Testing
@testable import VecturaOAIKit

@Suite("OpenAICompatibleEmbedder", .serialized)
struct OpenAICompatibleEmbedderTests {
  @Test("POSTs the expected payload and sorts vectors by index")
  func embedPostsExpectedPayloadAndSortsVectorsByIndex() async throws {
    MockURLProtocol.setHandler { request in
      #expect(request.url?.absoluteString == "http://localhost:1234/v1/embeddings")
      #expect(request.httpMethod == "POST")
      #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
      #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")

      let body = try #require(readRequestBody(from: request))
      let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
      #expect(json["model"] as? String == "text-embedding-3-small")
      #expect(json["input"] as? [String] == ["first", "second"])

      let url = try #require(request.url)
      let response = try #require(
        HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: nil,
          headerFields: nil
        )
      )
      let data = Data(
        """
        {
          "data": [
            { "index": 1, "embedding": [3.0, 4.0] },
            { "index": 0, "embedding": [1.0, 2.0] }
          ]
        }
        """.utf8
      )
      return (response, data)
    }
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder(apiKey: "secret", retryAttempts: 2)
    let embeddings = try await embedder.embed(texts: ["first", "second"])

    #expect(embeddings.count == 2)
    #expect(embeddings[0] == [1.0, 2.0])
    #expect(embeddings[1] == [3.0, 4.0])
  }

  @Test("Builds the embeddings endpoint from a trailing-slash base URL")
  func embedBuildsEndpointFromTrailingSlashBaseURL() async throws {
    MockURLProtocol.setHandler { request in
      #expect(request.url?.absoluteString == "http://localhost:1234/v1/embeddings")

      let url = try #require(request.url)
      let response = try #require(
        HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: nil,
          headerFields: nil
        )
      )
      let data = Data(#"{"data":[{"index":0,"embedding":[1.0,2.0]}]}"#.utf8)
      return (response, data)
    }
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder(baseURL: "http://localhost:1234/v1/")
    let embeddings = try await embedder.embed(texts: ["first"])

    #expect(embeddings == [[1.0, 2.0]])
  }

  @Test("Accepts a fully qualified embeddings endpoint without duplicating the path")
  func embedAcceptsFullyQualifiedEmbeddingsEndpoint() async throws {
    MockURLProtocol.setHandler { request in
      #expect(request.url?.absoluteString == "http://localhost:1234/custom/v1/embeddings")

      let url = try #require(request.url)
      let response = try #require(
        HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: nil,
          headerFields: nil
        )
      )
      let data = Data(#"{"data":[{"index":0,"embedding":[3.0,4.0]}]}"#.utf8)
      return (response, data)
    }
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder(baseURL: "http://localhost:1234/custom/v1/embeddings")
    let embeddings = try await embedder.embed(texts: ["first"])

    #expect(embeddings == [[3.0, 4.0]])
  }

  @Test("Caches the detected dimension after the first request")
  func dimensionIsCachedAfterFirstRequest() async throws {
    let requestCount = CounterBox()
    MockURLProtocol.setHandler { request in
      requestCount.value += 1

      let url = try #require(request.url)
      let response = try #require(
        HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: nil,
          headerFields: nil
        )
      )
      let data = Data(#"{"data":[{"index":0,"embedding":[1.0,2.0,3.0]}]}"#.utf8)
      return (response, data)
    }
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder()
    let first = try await embedder.dimension
    let second = try await embedder.dimension

    #expect(first == 3)
    #expect(second == 3)
    #expect(requestCount.value == 1)
  }

  @Test("Retries once after a rate-limit response")
  func embedRetriesAfterRateLimit() async throws {
    let requestCount = CounterBox()
    MockURLProtocol.setHandler { request in
      requestCount.value += 1

      let url = try #require(request.url)
      if requestCount.value == 1 {
        let response = try #require(
          HTTPURLResponse(
            url: url,
            statusCode: 429,
            httpVersion: nil,
            headerFields: ["Retry-After": "0.001"]
          )
        )
        return (response, Data("rate limited".utf8))
      }

      let response = try #require(
        HTTPURLResponse(
          url: url,
          statusCode: 200,
          httpVersion: nil,
          headerFields: nil
        )
      )
      let data = Data(#"{"data":[{"index":0,"embedding":[42.0]}]}"#.utf8)
      return (response, data)
    }
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder(retryAttempts: 1)
    let embeddings = try await embedder.embed(texts: ["retry me"])

    #expect(requestCount.value == 2)
    #expect(embeddings == [[42.0]])
  }

  @Test("Rejects whitespace-only input")
  func embedRejectsWhitespaceOnlyInput() async {
    defer { MockURLProtocol.setHandler(nil) }

    let embedder = makeEmbedder()

    await #expect(throws: OpenAICompatibleEmbedderError.self) {
      _ = try await embedder.embed(texts: ["   "])
    }
  }

  private func makeEmbedder(
    baseURL: String = "http://localhost:1234/v1",
    apiKey: String? = nil,
    retryAttempts: Int = 0
  ) -> OpenAICompatibleEmbedder {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: configuration)

    return OpenAICompatibleEmbedder(
      baseURL: baseURL,
      model: "text-embedding-3-small",
      apiKey: apiKey,
      timeoutInterval: 5,
      retryAttempts: retryAttempts,
      retryBaseDelaySeconds: 1,
      session: session
    )
  }
}

private final class MockURLProtocol: URLProtocol {
    private static let storage = RequestHandlerStorage()

    static func setHandler(_ handler: (@Sendable (URLRequest) throws -> (HTTPURLResponse, Data))?) {
        storage.set(handler)
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            let (response, data) = try Self.storage.handle(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private final class RequestHandlerStorage: @unchecked Sendable {
    private let lock = NSLock()
    private var requestHandler: (@Sendable (URLRequest) throws -> (HTTPURLResponse, Data))?

    func set(_ handler: (@Sendable (URLRequest) throws -> (HTTPURLResponse, Data))?) {
        lock.lock()
        defer { lock.unlock() }
        requestHandler = handler
    }

    func handle(_ request: URLRequest) throws -> (HTTPURLResponse, Data) {
        lock.lock()
        let requestHandler = requestHandler
        lock.unlock()

        guard let requestHandler else {
            throw NSError(domain: "MockURLProtocol", code: -1)
        }
        return try requestHandler(request)
    }
}

private final class CounterBox: @unchecked Sendable {
    var value = 0
}

private func readRequestBody(from request: URLRequest) -> Data? {
    if let httpBody = request.httpBody {
        return httpBody
    }

    guard let stream = request.httpBodyStream else {
        return nil
    }

    stream.open()
    defer { stream.close() }

    let bufferSize = 1024
    var data = Data()
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
    defer { buffer.deallocate() }

    while stream.hasBytesAvailable {
        let read = stream.read(buffer, maxLength: bufferSize)
        guard read > 0 else { break }
        data.append(buffer, count: read)
    }

    return data.isEmpty ? nil : data
}
