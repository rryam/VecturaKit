import Testing
@testable import VecturaKit

@Suite("SwiftEmbedder Resolution")
struct SwiftEmbedderResolutionTests {

  @Test("Explicit model type overrides heuristics")
  func explicitModelTypeOverridesHeuristics() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("minishlab/potion-base-4M", type: .bert)
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .bert)
  }

  @Test("Model2Vec family inferred from known ids")
  func inferModel2VecFamily() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("minishlab/potion-base-4M")
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .model2vec)
  }

  @Test("StaticEmbeddings family inferred from known ids")
  func inferStaticEmbeddingsFamily() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("sentence-transformers/static-retrieval-mrl-en-v1")
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .staticEmbeddings)
  }

  @Test("NomicBert family inferred from known ids")
  func inferNomicBertFamily() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("nomic-ai/nomic-embed-text-v1.5")
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .nomicBert)
  }

  @Test("ModernBert family inferred from known ids")
  func inferModernBertFamily() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("nomic-ai/modernbert-embed-base")
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .modernBert)
  }

  @Test("Explicit ModernBert type overrides heuristics")
  func explicitModernBertTypeOverridesHeuristics() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("sentence-transformers/all-MiniLM-L6-v2", type: .modernBert)
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .modernBert)
  }

  @Test("Unknown models default to BERT family")
  func unknownModelDefaultsToBertFamily() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let source = VecturaModelSource.id("sentence-transformers/all-MiniLM-L6-v2")
    let family = SwiftEmbedder.resolveModelFamily(for: source)
    #expect(family == .bert)
  }

  @Test("Static dimension uses base when truncate not set")
  func staticDimensionNoTruncate() throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let resolved = try SwiftEmbedder.resolvedStaticEmbeddingDimension(
      baseDimension: 768,
      truncateDimension: nil
    )
    #expect(resolved == 768)
  }

  @Test("Static dimension is truncated when requested")
  func staticDimensionTruncated() throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let resolved = try SwiftEmbedder.resolvedStaticEmbeddingDimension(
      baseDimension: 768,
      truncateDimension: 256
    )
    #expect(resolved == 256)
  }

  @Test("Static dimension caps truncate at base dimension")
  func staticDimensionCappedAtBase() throws {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    let resolved = try SwiftEmbedder.resolvedStaticEmbeddingDimension(
      baseDimension: 384,
      truncateDimension: 768
    )
    #expect(resolved == 384)
  }

  @Test("Static dimension rejects non-positive truncation")
  func staticDimensionRejectsInvalidTruncation() {
    guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
      return
    }
    #expect(throws: VecturaError.self) {
      _ = try SwiftEmbedder.resolvedStaticEmbeddingDimension(
        baseDimension: 384,
        truncateDimension: 0
      )
    }
  }
}
