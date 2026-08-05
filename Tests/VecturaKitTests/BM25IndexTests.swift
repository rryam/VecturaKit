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

  /// The textbook IDF, log((N - df + 0.5) / (df + 0.5)), is <= 0 whenever a term
  /// appears in at least half the corpus, and the `score > 0` filter then drops
  /// those documents. A single-document index is the extreme case: every term
  /// matches every document, so search returned nothing at all.
  @Test("Single-document index matches its own terms")
  func singleDocumentIndexMatchesOwnTerms() async {
    let doc = VecturaDocument(text: "hello world", embedding: [0.1])
    let index = BM25Index(documents: [doc], k1: 1.2, b: 0.75)

    let results = await index.search(query: "hello", topK: 10)

    #expect(results.map(\.document.id) == [doc.id])
    #expect((results.first?.score ?? 0) > 0)
  }

  /// A term in exactly half the corpus produced an IDF of exactly zero.
  @Test("Term appearing in half the corpus is still returned")
  func termInHalfTheCorpusIsReturned() async {
    let match = VecturaDocument(text: "banana grove", embedding: [0.1])
    let other = VecturaDocument(text: "apple orchard", embedding: [0.1])
    let index = BM25Index(documents: [match, other], k1: 1.2, b: 0.75)

    let results = await index.search(query: "banana", topK: 10)

    #expect(results.map(\.document.id) == [match.id])
  }

  /// A term in every document must still rank, and rarer terms must outrank it.
  @Test("Rarer terms outrank corpus-wide terms")
  func rarerTermsOutrankCommonTerms() async {
    let rare = VecturaDocument(text: "swift concurrency", embedding: [0.1])
    let common1 = VecturaDocument(text: "swift basics", embedding: [0.1])
    let common2 = VecturaDocument(text: "swift generics", embedding: [0.1])
    let index = BM25Index(documents: [rare, common1, common2], k1: 1.2, b: 0.75)

    // "swift" is in all three documents, so it must not be silently discarded.
    let commonResults = await index.search(query: "swift", topK: 10)
    #expect(commonResults.count == 3)
    #expect(commonResults.allSatisfy { $0.score > 0 })

    // The rarer term must score higher than the corpus-wide one.
    let rareResults = await index.search(query: "concurrency", topK: 10)
    let rareScore = try? #require(rareResults.first?.score)
    let commonScore = try? #require(commonResults.first?.score)
    #expect((rareScore ?? 0) > (commonScore ?? 0))
  }
}
