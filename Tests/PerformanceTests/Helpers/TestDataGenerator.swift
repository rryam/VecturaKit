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
}
