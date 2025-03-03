import Accelerate
import Foundation
import MLXEmbedders
import VecturaKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class VecturaMLXKit {
    private let config: VecturaConfig
    private let embedder: MLXEmbedder
    private var documents: [UUID: VecturaDocument] = [:]
    private var normalizedEmbeddings: [UUID: [Float]] = [:]
    private let storageDirectory: URL
    
    public init(config: VecturaConfig, modelConfiguration: ModelConfiguration = .nomic_text_v1_5)
    async throws
    {
        self.config = config
        self.embedder = try await MLXEmbedder(configuration: modelConfiguration)
        
        if let customStorageDirectory = config.directoryURL {
            let databaseDirectory = customStorageDirectory.appending(path: config.name)
            
            if !FileManager.default.fileExists(atPath: databaseDirectory.path(percentEncoded: false)) {
                try FileManager.default.createDirectory(
                    at: databaseDirectory, withIntermediateDirectories: true)
            }
            
            self.storageDirectory = databaseDirectory
        } else {
            // Create default storage directory
            self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first!
                .appendingPathComponent("VecturaKit")
                .appendingPathComponent(config.name)
        }
        
        try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
        
        // Attempt to load existing docs
        try loadDocuments()
    }
    
    public func addDocuments(texts: [String], ids: [UUID]? = nil) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }
        
        let embeddings = await embedder.embed(texts: texts)
        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()
        
        for (index, text) in texts.enumerated() {
            let docId = ids?[index] ?? UUID()
            let doc = VecturaDocument(id: docId, text: text, embedding: embeddings[index])
            
            // Normalize embedding for cosine similarity
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
            documentIds.append(docId)
            documentsToSave.append(doc)
        }
        
        try await withThrowingTaskGroup(of: Void.self) { group in
            let directory = self.storageDirectory
            
            for doc in documentsToSave {
                group.addTask {
                    let documentURL = directory.appendingPathComponent("\(doc.id).json")
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = .prettyPrinted
                    
                    let data = try encoder.encode(doc)
                    try data.write(to: documentURL)
                }
            }
            
            try await group.waitForAll()
        }
        
        return documentIds
    }
    
    public func search(query: String, numResults: Int? = nil, threshold: Float? = nil) async throws
    -> [VecturaSearchResult]
    {
        guard !query.isEmpty else {
            throw VecturaError.invalidInput("Query cannot be empty")
        }
        
        let queryEmbedding = try await embedder.embed(text: query)
        
        let norm = l2Norm(queryEmbedding)
        var divisorQuery = norm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(
            queryEmbedding, 1, &divisorQuery, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))
        
        var results: [VecturaSearchResult] = []
        
        for doc in documents.values {
            guard let normDoc = normalizedEmbeddings[doc.id] else { continue }
            let similarity = dotProduct(normalizedQuery, normDoc)
            
            if let minT = threshold ?? config.searchOptions.minThreshold, similarity < minT {
                continue
            }
            
            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: similarity,
                    createdAt: doc.createdAt
                )
            )
        }
        
        results.sort { $0.score > $1.score }
        
        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
    }
    
    public func deleteDocuments(ids: [UUID]) async throws {
        for id in ids {
            documents[id] = nil
            normalizedEmbeddings[id] = nil
            
            let documentURL = storageDirectory.appendingPathComponent("\(id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
    }
    
    public func updateDocument(id: UUID, newText: String) async throws {
        try await deleteDocuments(ids: [id])
        _ = try await addDocuments(texts: [newText], ids: [id])
    }
    
    public func reset() async throws {
        documents.removeAll()
        normalizedEmbeddings.removeAll()
        
        let files = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        for fileURL in files {
            try FileManager.default.removeItem(at: fileURL)
        }
    }
    
    // MARK: - Private
    
    private func loadDocuments() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        
        let decoder = JSONDecoder()
        var loadErrors: [String] = []
        
        for fileURL in fileURLs where fileURL.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let doc = try decoder.decode(VecturaDocument.self, from: data)
                
                // Rebuild normalized embeddings
                let norm = l2Norm(doc.embedding)
                var divisor = norm + 1e-9
                var normalized = [Float](repeating: 0, count: doc.embedding.count)
                vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
                normalizedEmbeddings[doc.id] = normalized
                documents[doc.id] = doc
            } catch {
                loadErrors.append(
                    "Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }
        
        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
        }
    }
    
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
    
    private func l2Norm(_ v: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
        return sqrt(sumSquares)
    }
}
