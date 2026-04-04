import Foundation
import XCTest
@testable import VecturaOAIKit

final class OpenAICompatibleEmbedderTests: XCTestCase {
    func testEmbedPostsExpectedPayloadAndSortsVectorsByIndex() async throws {
        MockURLProtocol.setHandler { request in
            XCTAssertEqual(request.url?.absoluteString, "http://localhost:1234/v1/embeddings")
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer secret")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")

            let body = try XCTUnwrap(readRequestBody(from: request))
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
            XCTAssertEqual(json["model"] as? String, "text-embedding-3-small")
            XCTAssertEqual(json["input"] as? [String], ["first", "second"])

            let response = HTTPURLResponse(
                url: try XCTUnwrap(request.url),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
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

        let embedder = OpenAICompatibleEmbedder(
            baseURL: "http://localhost:1234/v1",
            model: "text-embedding-3-small",
            apiKey: "secret",
            timeoutInterval: 5,
            retryAttempts: 2,
            retryBaseDelaySeconds: 1,
            session: makeSession()
        )

        let embeddings = try await embedder.embed(texts: ["first", "second"])

        XCTAssertEqual(embeddings.count, 2)
        XCTAssertEqual(embeddings[0], [1.0, 2.0])
        XCTAssertEqual(embeddings[1], [3.0, 4.0])
    }

    func testDimensionIsCachedAfterFirstRequest() async throws {
        let requestCount = CounterBox()
        MockURLProtocol.setHandler { request in
            requestCount.value += 1
            let response = HTTPURLResponse(
                url: try XCTUnwrap(request.url),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            let data = Data(#"{"data":[{"index":0,"embedding":[1.0,2.0,3.0]}]}"#.utf8)
            return (response, data)
        }
        defer { MockURLProtocol.setHandler(nil) }

        let embedder = OpenAICompatibleEmbedder(
            baseURL: "http://localhost:1234/v1",
            model: "text-embedding-3-small",
            timeoutInterval: 5,
            retryAttempts: 0,
            retryBaseDelaySeconds: 1,
            session: makeSession()
        )

        let first = try await embedder.dimension
        let second = try await embedder.dimension

        XCTAssertEqual(first, 3)
        XCTAssertEqual(second, 3)
        XCTAssertEqual(requestCount.value, 1)
    }

    func testEmbedRetriesAfterRateLimit() async throws {
        let requestCount = CounterBox()
        MockURLProtocol.setHandler { request in
            requestCount.value += 1

            if requestCount.value == 1 {
                let response = HTTPURLResponse(
                    url: try XCTUnwrap(request.url),
                    statusCode: 429,
                    httpVersion: nil,
                    headerFields: ["Retry-After": "0.001"]
                )!
                return (response, Data("rate limited".utf8))
            }

            let response = HTTPURLResponse(
                url: try XCTUnwrap(request.url),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            let data = Data(#"{"data":[{"index":0,"embedding":[42.0]}]}"#.utf8)
            return (response, data)
        }
        defer { MockURLProtocol.setHandler(nil) }

        let embedder = OpenAICompatibleEmbedder(
            baseURL: "http://localhost:1234/v1",
            model: "text-embedding-3-small",
            timeoutInterval: 5,
            retryAttempts: 1,
            retryBaseDelaySeconds: 1,
            session: makeSession()
        )

        let embeddings = try await embedder.embed(texts: ["retry me"])

        XCTAssertEqual(requestCount.value, 2)
        XCTAssertEqual(embeddings, [[42.0]])
    }

    func testEmbedRejectsWhitespaceOnlyInput() async {
        defer { MockURLProtocol.setHandler(nil) }

        let embedder = OpenAICompatibleEmbedder(
            baseURL: "http://localhost:1234/v1",
            model: "text-embedding-3-small",
            timeoutInterval: 5,
            retryAttempts: 0,
            retryBaseDelaySeconds: 1,
            session: makeSession()
        )

        do {
            _ = try await embedder.embed(texts: ["   "])
            XCTFail("Expected invalid input error")
        } catch let error as OpenAICompatibleEmbedderError {
            guard case .invalidInput(let message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("index 0"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: configuration)
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
