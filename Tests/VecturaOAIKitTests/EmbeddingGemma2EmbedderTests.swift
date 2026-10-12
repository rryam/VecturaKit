import Foundation
import Testing
import VecturaKit
@testable import VecturaOAIKit

@Suite("EmbeddingGemma2 unit tests (HTTP fixtures, no inference)", .serialized)
struct EmbeddingGemma2EmbedderTests {
  @Test("Pinned precision choices select distinct artifacts and server arguments",
    arguments: EmbeddingGemma2Configuration.Precision.allCases)
  func precisionSelection(_ precision: EmbeddingGemma2Configuration.Precision) throws {
    let config = try EmbeddingGemma2Configuration(precision: precision)
    let args = config.llamaServerArguments(modelDirectory: URL(filePath: "/tmp/models"), port: 19482)
    #expect(args[1] == "/tmp/models/" + precision.filename)
    #expect(args[3] == config.modelAlias)
    #expect(config.modelAlias.contains(precision.rawValue))
    #expect(config.precision.downloadURL.absoluteString.contains(EmbeddingGemma2Configuration.modelRevision))
    #expect(config.precision.sha256.count == 64)
    #expect(config.precision.artifactSizeBytes == (precision == .q8Zero ? 309_855_456 : 557_950_176))
    #expect(args.contains("mean"))
    #expect(args.contains("19482"))
    #expect(EmbeddingGemma2Configuration.Precision(rawValue: "Q4_0") == nil)
  }

  @Test("Image queries opt into a verified projector without changing the text embedding identity")
  func imageConfiguration() throws {
    for precision in EmbeddingGemma2Configuration.Precision.allCases {
      let text = try EmbeddingGemma2Configuration(precision: precision)
      let images = try EmbeddingGemma2Configuration(precision: precision, enableImageQueries: true)
      #expect(text.embeddingSpaceIdentifier == images.embeddingSpaceIdentifier)
      #expect(!text.llamaServerArguments(modelDirectory: URL(filePath: "/tmp")).contains("--mmproj"))
      let arguments = images.llamaServerArguments(modelDirectory: URL(filePath: "/tmp"))
      #expect(arguments.suffix(2) == ["--mmproj", "/tmp/" + precision.projectorFilename])
      #expect(precision.projectorSHA256.count == 64)
      #expect(precision.projectorSizeBytes == (precision == .q8Zero ? 554_821_024 : 982_074_784))
      #expect(precision.projectorDownloadURL.lastPathComponent == precision.projectorFilename)
    }
  }

  @Test("Image input uses nested media content without a text prompt", arguments: [true, false])
  func imageRequestShape(_ png: Bool) async throws {
    let config = try EmbeddingGemma2Configuration(dimensions: .d256, enableImageQueries: true)
    let bytes = Data(png ? [137, 80, 78, 71, 13, 10, 26, 10, 1] : [255, 216, 255, 1])
    let format: EmbeddingGemma2Embedder.ImageFormat = png ? .png : .jpeg
    let session = makeSession(config: config) { request in
      let body = try #require(requestBody(request))
      let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
      let inputs = try #require(json["input"] as? [[String: Any]])
      #expect(inputs.count == 1)
      let content = try #require(inputs[0]["content"] as? [[String: Any]])
      #expect(content.count == 1)
      #expect(content[0]["type"] as? String == "image_url")
      let image = try #require(content[0]["image_url"] as? [String: String])
      #expect(image["url"] == "data:" + format.rawValue + ";base64," + bytes.base64EncodedString())
      return try response(request, vectors: [(0, Array(repeating: 1, count: 768))])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let vector = try await embedder.embedImageQuery(data: bytes, format: format)
    #expect(vector.count == 256)
    #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 0.00001)
  }

  @Test("Disabled, empty, wrong-format and oversized images fail before HTTP")
  func invalidImages() async throws {
    let disabled = try EmbeddingGemma2Embedder(configuration: .init(), baseURL: "http://127.0.0.1:1/v1")
    let signature = Data([137, 80, 78, 71, 13, 10, 26, 10])
    await #expect(throws: OpenAICompatibleEmbedderError.self) {
      try await disabled.embedImageQuery(data: signature, format: .png)
    }
    let enabled = try EmbeddingGemma2Embedder(
      configuration: .init(enableImageQueries: true), baseURL: "http://127.0.0.1:1/v1"
    )
    for bytes in [Data(), Data("caption".utf8), Data(repeating: 0, count: 10 * 1024 * 1024 + 1)] {
      await #expect(throws: OpenAICompatibleEmbedderError.self) {
        try await enabled.embedImageQuery(data: bytes, format: .png)
      }
    }
    await #expect(throws: OpenAICompatibleEmbedderError.self) {
      try await enabled.embedImageQuery(data: signature, format: .jpeg)
    }
  }

  @Test("Image queries reject a text-only server before uploading bytes")
  func imageCapabilityRequired() async throws {
    let config = try EmbeddingGemma2Configuration(enableImageQueries: true)
    let session = makeSession(config: config, imageCapable: false) { _ in
      Issue.record("A text-only server must not receive image bytes")
      throw URLError(.badServerResponse)
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: OpenAICompatibleEmbedderError.self) {
      try await embedder.embedImageQuery(data: Data([255, 216, 255]), format: .jpeg)
    }
  }

  @Test("Image queries propagate server errors and reject malformed vectors", arguments: [0, 1, 2])
  func imageFailures(_ fixture: Int) async throws {
    let config = try EmbeddingGemma2Configuration(enableImageQueries: true)
    let session = makeSession(config: config) { request in
      if fixture == 0 { return (try httpResponse(request, status: 400), Data("Invalid image".utf8)) }
      return try response(request, vectors: [(0, Array(repeating: 0, count: fixture == 1 ? 767 : 768))])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: (any Error).self) {
      try await embedder.embedImageQuery(data: Data([255, 216, 255]), format: .jpeg)
    }
  }

  @Test("Cancellation during an image request returns no vector")
  func imageCancellation() async throws {
    let config = try EmbeddingGemma2Configuration(enableImageQueries: true)
    let started = GemmaCalls()
    let session = makeSession(config: config) { request in
      started.append(["started"])
      try await Task.sleep(for: .seconds(30))
      return try response(request, vectors: [(0, Array(repeating: 1, count: 768))])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let task = Task { try await embedder.embedImageQuery(data: Data([255, 216, 255]), format: .jpeg) }
    for _ in 0..<100 where started.values.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
    #expect(!started.values.isEmpty)
    task.cancel()
    await #expect(throws: (any Error).self) { try await task.value }
  }

  @Test("Invalid configuration fails before any HTTP request")
  func invalidConfiguration() throws {
    for size in [-1, 0, 33, Int.max] {
      #expect(throws: OpenAICompatibleEmbedderError.self) {
        try EmbeddingGemma2Configuration(batchSize: size)
      }
    }
    let config = try EmbeddingGemma2Configuration()
    for url in ["", "file:///tmp/model", "localhost", "http://localhost/v1?x=1"] {
      #expect(throws: OpenAICompatibleEmbedderError.self) {
        try EmbeddingGemma2Embedder(configuration: config, baseURL: url)
      }
    }
    #expect(throws: OpenAICompatibleEmbedderError.self) {
      try EmbeddingGemma2Embedder(configuration: config, timeoutInterval: .infinity)
    }
  }

  @Test("Document/query formatting, bounded batching and response order")
  func promptsAndBatches() async throws {
    let config = try EmbeddingGemma2Configuration(dimensions: .d128, batchSize: 2)
    let calls = GemmaCalls()
    let session = makeSession(config: config) { request in
      let inputs = try requestInputs(request)
      calls.append(inputs)
      return try response(request, vectors: inputs.enumerated().reversed().map { index, input in
        let value: Double = input.contains("second") ? 4 : 3
        return (index, [value, 4] + Array(repeating: 0, count: 766))
      })
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let vectors = try await embedder.embed(texts: ["first", "second", "third"])
    let query = try await embedder.embedQuery(text: "where?")
    #expect(calls.values == [
      ["title: none | text: first", "title: none | text: second"],
      ["title: none | text: third"], ["task: search result | query: where?"]
    ])
    #expect(abs(vectors[0][0] - 0.6) < 0.00001)
    #expect(abs(vectors[1][0] - Float(1 / sqrt(2.0))) < 0.00001)
    #expect(vectors[2] == vectors[0])
    #expect(query.count == 128)
    #expect(try await embedder.dimension == 128)
  }

  @Test("Each Matryoshka dimension is truncated then renormalized",
    arguments: EmbeddingGemma2Configuration.Dimensions.allCases)
  func dimensions(_ dimensions: EmbeddingGemma2Configuration.Dimensions) async throws {
    let config = try EmbeddingGemma2Configuration(dimensions: dimensions)
    let session = makeSession(config: config) { request in
      var vector = [Double](repeating: 0, count: 768)
      vector[0] = 3
      vector[127] = 4
      vector[767] = 12
      return try response(request, vectors: [(0, vector)])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let vector = try await embedder.embed(text: "document")
    #expect(vector.count == dimensions.rawValue)
    #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 0.00001)
    #expect(abs(vector[0] - (dimensions == .d768 ? 3.0 / 13 : 0.6)) < 0.00001)
  }

  @Test("Rejects empty and whitespace inputs without HTTP")
  func invalidInputs() async throws {
    let embedder = try EmbeddingGemma2Embedder(configuration: .init(), baseURL: "http://127.0.0.1:1/v1")
    for inputs in [[], [""], [" \n"], ["valid", ""]] {
      await #expect(throws: OpenAICompatibleEmbedderError.self) { try await embedder.embed(texts: inputs) }
    }
    await #expect(throws: OpenAICompatibleEmbedderError.self) { try await embedder.embedQuery(text: " ") }
  }

  @Test("Rejects wrong dimensions, zero and nonfinite vectors", arguments: [0, 1, 2, 3])
  func invalidVectors(_ fixture: Int) async throws {
    let config = try EmbeddingGemma2Configuration(dimensions: .d128)
    let session = makeSession(config: config) { request in
      var vector = [Double](repeating: 0, count: fixture == 0 ? 767 : 768)
      if fixture == 2 { vector[767] = 1 } // Zero norm after truncation.
      if fixture == 3 { vector[0] = Double.greatestFiniteMagnitude } // Overflows Float.
      return try response(request, vectors: [(0, vector)])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: (any Error).self) { try await embedder.embed(text: "document") }
  }

  @Test("Rejects missing, duplicate and out-of-range response indices", arguments: [0, 1, 2, 3, 4, 5, 6])
  func invalidIndices(_ fixture: Int) async throws {
    let config = try EmbeddingGemma2Configuration()
    let session = makeSession(config: config) { request in
      let vector = [Double](repeating: 1, count: 768)
      let pairs = fixture == 0 ? [(0, vector)] : [(0, vector), (fixture == 1 ? 0 : 2, vector)]
      if fixture >= 4 {
        var items: [[String: Any]] = [["index": 0, "embedding": vector], ["index": 1, "embedding": vector]]
        if fixture == 4 { items.append(["ignored": true]) }
        if fixture == 5 { items[1]["index"] = true }
        if fixture == 6 { items[1]["index"] = 1.5 }
        let data = try JSONSerialization.data(withJSONObject: ["data": items])
        return (try httpResponse(request), data)
      }
      if fixture == 3 {
        let data = try JSONSerialization.data(withJSONObject: ["data": [["embedding": vector]]])
        return (try httpResponse(request), data)
      }
      return try response(request, vectors: pairs)
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: OpenAICompatibleEmbedderError.self) {
      try await embedder.embed(texts: ["one", "two"])
    }
  }

  @Test("HTTP and transport failures propagate", arguments: [0, 1])
  func failures(_ fixture: Int) async throws {
    let config = try EmbeddingGemma2Configuration()
    let session = makeSession(config: config) { request in
      if fixture == 0 { return (try httpResponse(request, status: 503), Data()) }
      throw URLError(.cannotConnectToHost)
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: (any Error).self) { try await embedder.embed(text: "document") }
  }

  @Test("Rejects a server exposing a different precision alias")
  func modelMismatch() async throws {
    let config = try EmbeddingGemma2Configuration()
    let session = makeSession(config: try .init(precision: .bf16)) { _ in
      Issue.record("Mismatched server must not receive embedding requests")
      throw URLError(.badServerResponse)
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    await #expect(throws: OpenAICompatibleEmbedderError.self) { try await embedder.embed(text: "document") }
  }

  @Test("Cancellation before inference prevents requests")
  func cancellationBeforeRequest() async throws {
    let embedder = try EmbeddingGemma2Embedder(configuration: .init(), baseURL: "http://127.0.0.1:1/v1")
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return try await embedder.embed(text: "document")
    }
    await #expect(throws: CancellationError.self) { try await task.value }
  }

  @Test("Cancellation during HTTP cancels inference and returns no vectors")
  func cancellationDuringRequest() async throws {
    let config = try EmbeddingGemma2Configuration()
    let started = GemmaCalls()
    let session = makeSession(config: config) { request in
      started.append(["started"])
      try await Task.sleep(for: .seconds(30))
      return try response(request, vectors: [(0, Array(repeating: 1, count: 768))])
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let task = Task { try await embedder.embed(text: "document") }
    for _ in 0..<100 where started.values.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
    #expect(!started.values.isEmpty)
    task.cancel()
    await #expect(throws: (any Error).self) { try await task.value }
  }

  @Test("Database add, update, hybrid and vector search dispatch the appropriate prompts")
  func databaseDispatch() async throws {
    let config = try EmbeddingGemma2Configuration(dimensions: .d128)
    let calls = GemmaCalls()
    let session = makeSession(config: config) { request in
      let inputs = try requestInputs(request)
      calls.append(inputs)
      return try response(request, vectors: inputs.indices.map { ($0, Array(repeating: 1, count: 768)) })
    }
    defer { session.invalidateAndCancel(); GemmaURLProtocol.handler.set(nil) }
    let directory = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let embedder = try EmbeddingGemma2Embedder(configuration: config, session: session)
    let database = try await VecturaKit(
      config: VecturaConfig(name: "test", directoryURL: directory), embedder: embedder
    )
    let identifier = try await database.addDocument(text: "original")
    try await database.updateDocument(id: identifier, newText: "updated")
    _ = try await database.search(query: "hybrid")
    let storage = try FileStorageProvider(
      storageDirectory: directory.appendingPathComponent("test"),
      embeddingSpaceIdentifier: config.embeddingSpaceIdentifier
    )
    _ = try await VectorSearchEngine(embedder: embedder).search(
      query: .text("vector"), storage: storage, options: SearchOptions()
    )
    #expect(calls.values == [["title: none | text: original"], ["title: none | text: updated"],
      ["task: search result | query: hybrid"], ["task: search result | query: vector"]])
  }
}

private typealias GemmaHandler = @Sendable (URLRequest) async throws -> (HTTPURLResponse, Data)

private final class GemmaURLProtocol: URLProtocol, @unchecked Sendable {
  static let handler = GemmaHandlerStorage()
  private var work: Task<Void, Never>?

  // swiftlint:disable:next static_over_final_class
  override class func canInit(with request: URLRequest) -> Bool { true }
  // swiftlint:disable:next static_over_final_class
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    work = Task {
      do {
        let (response, data) = try await Self.handler.handle(request)
        try Task.checkCancellation()
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
      } catch {
        client?.urlProtocol(self, didFailWithError: error)
      }
    }
  }

  override func stopLoading() { work?.cancel() }
}

private final class GemmaHandlerStorage: @unchecked Sendable {
  private let lock = NSLock()
  private var handler: GemmaHandler?
  func set(_ value: GemmaHandler?) { lock.withLock { handler = value } }
  func handle(_ request: URLRequest) async throws -> (HTTPURLResponse, Data) {
    let value: GemmaHandler? = lock.withLock { self.handler }
    let handler = try #require(value)
    return try await handler(request)
  }
}

private final class GemmaCalls: @unchecked Sendable {
  private let lock = NSLock()
  private var calls: [[String]] = []
  var values: [[String]] { lock.withLock { calls } }
  func append(_ inputs: [String]) { lock.withLock { calls.append(inputs) } }
}

private func makeSession(
  config: EmbeddingGemma2Configuration, imageCapable: Bool? = nil, embeddings: @escaping GemmaHandler
) -> URLSession {
  GemmaURLProtocol.handler.set { request in
    if request.url?.lastPathComponent == "models" {
      let modalities = (imageCapable ?? config.enableImageQueries) ? ["text", "image"] : ["text"]
      let data = try JSONSerialization.data(withJSONObject: ["data": [[
        "id": config.modelAlias, "architecture": ["input_modalities": modalities]
      ]]])
      return (try httpResponse(request), data)
    }
    #expect(request.value(forHTTPHeaderField: "Authorization") == nil)
    let body = try #require(requestBody(request))
    let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
    #expect(json["model"] as? String == config.modelAlias)
    return try await embeddings(request)
  }
  let configuration = URLSessionConfiguration.ephemeral
  configuration.protocolClasses = [GemmaURLProtocol.self]
  return URLSession(configuration: configuration)
}

private func httpResponse(_ request: URLRequest, status: Int = 200) throws -> HTTPURLResponse {
  let url = try #require(request.url)
  return try #require(HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil))
}

private func response(_ request: URLRequest, vectors: [(Int, [Double])]) throws -> (HTTPURLResponse, Data) {
  let data = try JSONSerialization.data(withJSONObject: ["data": vectors.map { ["index": $0.0, "embedding": $0.1] }])
  return (try httpResponse(request), data)
}

private func requestInputs(_ request: URLRequest) throws -> [String] {
  let body = try #require(requestBody(request))
  let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
  return try #require(json["input"] as? [String])
}

private func requestBody(_ request: URLRequest) -> Data? {
  if let body = request.httpBody { return body }
  guard let stream = request.httpBodyStream else { return nil }
  stream.open()
  defer { stream.close() }
  var data = Data()
  var buffer = [UInt8](repeating: 0, count: 1024)
  while stream.hasBytesAvailable {
    let count = stream.read(&buffer, maxLength: buffer.count)
    guard count > 0 else { break }
    data.append(contentsOf: buffer.prefix(count))
  }
  return data
}
