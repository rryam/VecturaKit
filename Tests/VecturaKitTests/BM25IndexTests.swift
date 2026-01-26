import Foundation
import Testing
@testable import VecturaKit

@Suite("BM25Index")
struct BM25IndexTests {

  @Test("Empty query returns no results")
  func emptyQueryReturnsNoResults() async {
    let doc = VecturaDocument(
      id: UUID(),
      text: "hello world",
      embedding: [0.1]
    )
    let index = BM25Index(documents: [doc], k1: 1.2, b: 0.75)

    let results = await index.search(query: "", topK: 10)

    #expect(results.isEmpty)
  }
}
