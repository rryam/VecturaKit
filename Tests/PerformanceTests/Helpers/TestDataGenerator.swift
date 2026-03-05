import Foundation

/// Generates test data for performance benchmarks.
public struct TestDataGenerator: Sendable {

  /// Predefined topics for generating diverse test documents.
  private static let topics = [
    "machine learning",
    "deep learning",
    "neural networks",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "data science",
    "algorithm optimization",
    "distributed systems",
    "cloud computing",
    "software architecture",
    "database design",
    "web development",
    "mobile applications",
    "cybersecurity",
    "blockchain technology",
    "quantum computing",
    "robotics and automation",
    "internet of things",
    "edge computing"
  ]

  /// Predefined context phrases for generating realistic sentences.
  private static let contexts = [
    "is a fundamental concept in",
    "plays a crucial role in",
    "has revolutionized the field of",
    "provides innovative solutions for",
    "enables efficient processing of",
    "improves the performance of",
    "offers new approaches to",
    "addresses key challenges in",
    "transforms the way we handle",
    "optimizes the workflow for"
  ]

  /// Predefined domains for context.
  private static let domains = [
    "technology and innovation",
    "modern software development",
    "data-driven applications",
    "enterprise solutions",
    "scientific research",
    "industrial automation",
    "digital transformation",
    "business intelligence",
    "healthcare systems",
    "financial services"
  ]

  private static let noisyTokens = [
    "latency",
    "throughput",
    "p95",
    "p99",
    "cache-hit",
    "cache-miss",
    "retry",
    "timeout",
    "batch",
    "vector-db",
    "k8s",
    "grpc",
    "ssd",
    "cold-start",
    "hot-path",
    "rollback"
  ]

  public init() {}

  /// Generate a collection of test documents.
  ///
  /// - Parameters:
  ///   - count: Number of documents to generate
  ///   - seed: Random seed for reproducibility (optional)
  /// - Returns: Array of test document texts
  public func generateDocuments(count: Int, seed: UInt64? = nil) -> [String] {
    var generator = seed.map { SeededRandomGenerator(seed: $0) } ?? SeededRandomGenerator(seed: 12345)

    return (0..<count).map { index in
      let topic = Self.topics[generator.next() % Self.topics.count]
      let context = Self.contexts[generator.next() % Self.contexts.count]
      let domain = Self.domains[generator.next() % Self.domains.count]

      return "Document \(index): \(topic) \(context) \(domain)."
    }
  }

  /// Generate query strings that should match various documents.
  ///
  /// - Parameters:
  ///   - count: Number of queries to generate
  ///   - seed: Random seed for reproducibility (optional)
  /// - Returns: Array of query strings
  public func generateQueries(count: Int, seed: UInt64? = nil) -> [String] {
    var generator = seed.map { SeededRandomGenerator(seed: $0) } ?? SeededRandomGenerator(seed: 54321)

    return (0..<count).map { _ in
      Self.topics[generator.next() % Self.topics.count]
    }
  }

  /// Generate a large dataset with specified characteristics.
  ///
  /// - Parameters:
  ///   - documentCount: Total number of documents
  ///   - avgWordsPerDoc: Average words per document (adds variability)
  ///   - seed: Random seed for reproducibility
  /// - Returns: Array of generated documents
  public func generateLargeDataset(
    documentCount: Int,
    avgWordsPerDoc: Int = 20,
    seed: UInt64? = nil
  ) -> [String] {
    var generator = seed.map { SeededRandomGenerator(seed: $0) } ?? SeededRandomGenerator(seed: 98765)

    let topicWords = Self.topics.flatMap { $0.split(separator: " ").map(String.init) }
    let contextWords = Self.contexts.flatMap { $0.split(separator: " ").map(String.init) }
    let domainWords = Self.domains.flatMap { $0.split(separator: " ").map(String.init) }
    let words = topicWords + contextWords + domainWords

    return (0..<documentCount).map { index in
      let wordCount = max(5, avgWordsPerDoc + Int(generator.next() % 20) - 10)
      let docWords = (0..<wordCount).map { _ in
        words[generator.next() % words.count]
      }
      return "Doc\(index): " + docWords.joined(separator: " ")
    }
  }

  /// Generate a corpus with realistic variance in document length and lexical noise.
  ///
  /// The output mixes:
  /// - short and long technical documents
  /// - repeated near-duplicates
  /// - numeric/error-code style tokens and punctuation
  ///
  /// - Parameters:
  ///   - count: Number of documents to generate
  ///   - minWords: Minimum words in a document
  ///   - maxWords: Maximum words in a document
  ///   - duplicateRate: Fraction of documents that are near-duplicates [0, 1]
  ///   - seed: Seed for reproducibility
  /// - Returns: Generated realistic corpus
  public func generateRealisticCorpus(
    count: Int,
    minWords: Int = 20,
    maxWords: Int = 260,
    duplicateRate: Double = 0.08,
    seed: UInt64? = nil
  ) -> [String] {
    guard count > 0 else { return [] }

    let clampedMinWords = max(5, minWords)
    let clampedMaxWords = max(clampedMinWords, maxWords)
    let clampedDuplicateRate = min(max(duplicateRate, 0), 1)

    var generator = seed.map { SeededRandomGenerator(seed: $0) } ?? SeededRandomGenerator(seed: 424242)
    let topicWords = Self.topics.flatMap { $0.split(separator: " ").map(String.init) }
    let contextWords = Self.contexts.flatMap { $0.split(separator: " ").map(String.init) }
    let domainWords = Self.domains.flatMap { $0.split(separator: " ").map(String.init) }
    let vocabulary = topicWords + contextWords + domainWords + Self.noisyTokens

    var corpus: [String] = []
    corpus.reserveCapacity(count)

    for index in 0..<count {
      let shouldDuplicate = !corpus.isEmpty && generator.nextDouble() < clampedDuplicateRate
      if shouldDuplicate {
        let base = corpus[generator.nextInt(upperBound: corpus.count)]
        let variant = base
          + " Variant-\(index) incident=\(1000 + generator.nextInt(upperBound: 9000))"
          + " status=\(generator.pick(from: ["ok", "warn", "error"]))"
        corpus.append(variant)
        continue
      }

      let topic = generator.pick(from: Self.topics)
      let context = generator.pick(from: Self.contexts)
      let domain = generator.pick(from: Self.domains)
      let wordCount = clampedMinWords + generator.nextInt(upperBound: clampedMaxWords - clampedMinWords + 1)

      var words: [String] = []
      words.reserveCapacity(wordCount)
      for position in 0..<wordCount {
        var token = vocabulary[generator.nextInt(upperBound: vocabulary.count)]
        if position % 37 == 0 {
          token += "-\(generator.nextInt(upperBound: 500))"
        }
        words.append(token)
      }

      let punctuation = generator.pick(from: [".", ".", ".", ";", "!", "?"])
      let body = words.joined(separator: " ")
      let record =
        "Document \(index): \(topic) \(context) \(domain). \(body)\(punctuation) err=\(generator.nextInt(upperBound: 12))"
      corpus.append(record)
    }

    return corpus
  }

  /// Generate realistic query traffic with short, medium, and long queries.
  ///
  /// - Parameters:
  ///   - count: Number of queries
  ///   - seed: Seed for reproducibility
  /// - Returns: Array of query strings
  public func generateRealisticQueries(
    count: Int,
    seed: UInt64? = nil
  ) -> [String] {
    guard count > 0 else { return [] }

    var generator = seed.map { SeededRandomGenerator(seed: $0) } ?? SeededRandomGenerator(seed: 898989)
    let tokens = Self.topics + Self.domains + Self.noisyTokens
    var queries: [String] = []
    queries.reserveCapacity(count)

    for _ in 0..<count {
      let mode = generator.nextInt(upperBound: 100)
      if mode < 40 {
        queries.append(generator.pick(from: Self.topics))
      } else if mode < 85 {
        let first = generator.pick(from: tokens)
        let second = generator.pick(from: tokens)
        let third = generator.pick(from: tokens)
        queries.append("\(first) \(second) \(third)")
      } else {
        let first = generator.pick(from: Self.topics)
        let second = generator.pick(from: Self.domains)
        let third = generator.pick(from: Self.noisyTokens)
        queries.append("how to optimize \(first) for \(second) with \(third)")
      }
    }

    return queries
  }
}

/// Simple seeded random number generator for reproducible tests.
private struct SeededRandomGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    self.state = seed
  }

  /// Generate next random number using LCG algorithm.
  mutating func next() -> Int {
    // Linear Congruential Generator
    state = state &* 6364136223846793005 &+ 1442695040888963407
    return Int(state >> 32)
  }

  mutating func nextInt(upperBound: Int) -> Int {
    guard upperBound > 0 else { return 0 }
    return next() % upperBound
  }

  mutating func nextDouble() -> Double {
    let value = UInt64(nextInt(upperBound: Int(UInt32.max)))
    return Double(value) / Double(UInt32.max)
  }

  mutating func pick<T>(from values: [T]) -> T {
    values[nextInt(upperBound: values.count)]
  }
}
