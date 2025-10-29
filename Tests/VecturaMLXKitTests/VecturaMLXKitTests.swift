import Foundation
import Metal
import MLX
import Testing
@testable import VecturaMLXKit
@testable import VecturaKit

/// Tests for VecturaKit with MLX embeddings functionality
///
/// Note: These tests require:
/// 1. Metal device (GPU) availability
/// 2. Metal Toolchain (install via: xcodebuild -downloadComponent MetalToolchain)
/// 3. MLX device libraries to be available
///
/// Run tests with: xcodebuild test -scheme VecturaMLXKitTests -destination 'platform=macOS'
/// (swift test may not work due to Metal library compilation requirements)
@Suite("VecturaMLXKit")
struct VecturaMLXKitTests {
    private let testDimension = 768

    private var shouldRunMLXTests: Bool {
        ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] == "1"
    }

    private var defaultSearchOptions: VecturaConfig.SearchOptions {
        .init(
            defaultNumResults: 10,
            minThreshold: 0,
            hybridWeight: 0.5,
            k1: 1.2,
            b: 0.75
        )
    }

    private func makeTestDirectory() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
        let directory = temp.appendingPathComponent(
            "VecturaMLXKitTests-\(UUID().uuidString)",
            isDirectory: true
        )

        if FileManager.default.fileExists(atPath: directory.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: directory)
        }

        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o755]
        )

        return directory
    }

    private func makeConfig(name: String = UUID().uuidString, directoryURL: URL) -> VecturaConfig {
        VecturaConfig(
            name: name,
            directoryURL: directoryURL,
            dimension: testDimension,
            searchOptions: defaultSearchOptions
        )
    }

    private func createVecturaKit(config: VecturaConfig) async throws -> VecturaKit? {
        guard shouldRunMLXTests else {
            return nil
        }

        guard MTLCreateSystemDefaultDevice() != nil else {
            return nil
        }

        do {
            let embedder = try await MLXEmbedder(configuration: .nomic_text_v1_5)
            return try await VecturaKit(config: config, embedder: embedder)
        } catch {
            print("Skipping MLX test due to initialization failure: \(error.localizedDescription)")
            return nil
        }
    }

    @Test("Add and search")
    func addAndSearch() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let text = "Hello world"
        let ids = try await kit.addDocuments(texts: [text])
        #expect(ids.count == 1, "Should add exactly one document.")

        let results = try await kit.search(query: text)
        #expect(results.count == 1, "The search should return one result after adding one document.")
        #expect(results.first?.text == text, "The text of the returned document should match the added text.")
    }

    @Test("Delete documents")
    func deleteDocuments() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let text = "Delete me"
        let ids = try await kit.addDocuments(texts: [text])
        #expect(ids.count == 1, "Should add exactly one document.")

        try await kit.deleteDocuments(ids: ids)

        let results = try await kit.search(query: text)
        #expect(results.isEmpty, "After deletion, the document should not be returned in search results.")
    }

    @Test("Update document")
    func updateDocument() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let originalText = "Original text"
        let updatedText = "Updated text"
        let ids = try await kit.addDocuments(texts: [originalText])
        #expect(ids.count == 1, "Should add exactly one document.")

        let documentID = try #require(ids.first)
        try await kit.updateDocument(id: documentID, newText: updatedText)

        let results = try await kit.search(query: updatedText)
        #expect(results.count == 1, "One document should be returned after update.")
        #expect(results.first?.text == updatedText, "The document text should be updated in the search results.")
    }

    @Test("Reset removes documents")
    func reset() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        _ = try await kit.addDocuments(texts: ["Doc1", "Doc2"])
        try await kit.reset()

        let results = try await kit.search(query: "Doc")
        #expect(results.isEmpty, "After a reset, search should return no results.")
    }

    @Test("Search multiple documents")
    func searchMultipleDocuments() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(name: "TestMLXDB", directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over lazy hounds",
            "An agile brown fox",
            "Lazy dogs sleep all day",
            "Quick and nimble foxes"
        ]
        _ = try await kit.addDocuments(texts: texts)

        let results = try await kit.search(query: "brown fox")
        #expect(results.count >= 2, "Should return at least two documents related to 'brown fox'.")

        for index in 1..<results.count {
            #expect(
                results[index - 1].score >= results[index].score,
                "Search results are not sorted in descending order by score."
            )
        }
    }

    @Test("Search result limiting")
    func searchNumResultsLimiting() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(name: "TestMLXDB", directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let texts = [
            "Document one about testing",
            "Document two about testing",
            "Document three about testing",
            "Document four about testing",
            "Document five about testing"
        ]
        _ = try await kit.addDocuments(texts: texts)

        let results = try await kit.search(query: "testing", numResults: 3)
        #expect(results.count == 3, "Should limit the search results to exactly 3 documents.")
    }

    @Test("Search high threshold")
    func searchWithHighThreshold() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(name: "TestMLXDB", directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        let texts = [
            "Apple pie recipe",
            "Delicious apple tart",
            "Banana bread instructions"
        ]
        _ = try await kit.addDocuments(texts: texts)

        let highThreshold: Float = 0.99
        let results = try await kit.search(query: "apple", threshold: highThreshold)

        for result in results {
            #expect(
                result.score >= highThreshold,
                "Result score \(result.score) is below the high threshold \(highThreshold)."
            )
        }
    }

    @Test("Search no matches")
    func searchNoMatches() async throws {
        let directory = try makeTestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = makeConfig(name: "TestMLXDB", directoryURL: directory)
        guard let kit = try await createVecturaKit(config: config) else { return }

        _ = try await kit.addDocuments(texts: ["Some random content"])

        let results = try await kit.search(query: "completely different query text", threshold: 0.9)
        #expect(results.isEmpty, "Search should return no results when the query does not match any document.")
    }
}
