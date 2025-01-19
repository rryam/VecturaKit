struct BM25Index {
    private let k1: Double
    private let b: Double
    private var documents: [Document]
    private var documentFrequencies: [String: Int]
    private var documentLengths: [String: Int]
    private let averageDocumentLength: Double
    
    init(documents: [Document], k1: Double = 1.2, b: Double = 0.75) {
        self.k1 = k1
        self.b = b
        self.documents = documents
        
        // Pre-calculate document lengths
        self.documentLengths = documents.reduce(into: [:]) { dict, doc in
            dict[doc.id] = tokenize(doc.content).count
        }
        
        // Calculate average document length
        self.averageDocumentLength = Double(documentLengths.values.reduce(0, +)) / Double(documents.count)
        
        // Pre-calculate document frequencies
        self.documentFrequencies = [:]
        for document in documents {
            let terms = Set(tokenize(document.content))
            for term in terms {
                documentFrequencies[term, default: 0] += 1
            }
        }
    }
    
    func search(query: String, topK: Int = 10) -> [(document: Document, score: Double)] {
        let queryTerms = tokenize(query)
        var scores: [(Document, Double)] = []
        
        for document in documents {
            let docLength = Double(documentLengths[document.id] ?? 0)
            var score = 0.0
            
            for term in queryTerms {
                let tf = termFrequency(term: term, in: document)
                let df = Double(documentFrequencies[term] ?? 0)
                
                // BM25 scoring formula
                let idf = log((Double(documents.count) - df + 0.5) / (df + 0.5))
                let numerator = tf * (k1 + 1)
                let denominator = tf + k1 * (1 - b + b * docLength / averageDocumentLength)
                
                score += idf * (numerator / denominator)
            }
            
            scores.append((document, score))
        }
        
        return scores
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .filter { $0.1 > 0 }
    }
    
    private func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .folding(options: .diacriticInsensitive, locale: .current)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
    
    private func termFrequency(term: String, in document: Document) -> Double {
        Double(tokenize(document.content)
            .filter { $0 == term }
            .count)
    }
}
