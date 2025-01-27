import CoreML
import Embeddings
import Foundation

/// A vector database implementation that stores and searches documents using their vector embeddings.
public class VecturaKit: VecturaProtocol {

    /// The configuration for this vector database instance.
    private let config: VecturaConfig

    /// The storage for documents.
    private var documents: [UUID: VecturaDocument]

    /// The storage directory for documents.
    private let storageDirectory: URL

    /// Cached normalized embeddings for faster searches.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]

    /// BM25 index for text search
    private var bm25Index: BM25Index?

    /// Swift-Embeddings model bundle that you can reuse (e.g. BERT, XLM-R, CLIP, etc.)
    private var bertModel: Bert.ModelBundle?

    // MARK: - Initialization

    public init(config: VecturaConfig) throws {
        self.config = config
        self.documents = [:]

        // Create storage directory
        self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("VecturaKit")
            .appendingPathComponent(config.name)

        try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)

        // Attempt to load existing docs
        try loadDocuments()
    }

    /// Adds multiple documents to the vector store in batch.
    public func addDocuments(
        texts: [String],
        ids: [UUID]? = nil,
        model: VecturaModelSource = .default
    ) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }

        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: model)
        }

        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(model)")
        }

        let embeddingsTensor = try modelBundle.batchEncode(texts)
        let shape = embeddingsTensor.shape

        if shape.count != 2 {
            throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
        }

        if shape[1] != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: shape[1]
            )
        }

        let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(
            of: Float.self)
        let allScalars = embeddingShapedArray.scalars

        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()

        for i in 0..<texts.count {
            let startIndex = i * config.dimension
            let endIndex = startIndex + config.dimension
            let embeddingRow = Array(allScalars[startIndex..<endIndex])

            let docId = ids?[i] ?? UUID()
            let doc = VecturaDocument(
                id: docId,
                text: texts[i],
                embedding: embeddingRow
            )
            documentsToSave.append(doc)
            documentIds.append(docId)
        }

        for doc in documentsToSave {
            let norm = l2Norm(doc.embedding)
            let normalized = doc.embedding.map { $0 / (norm + 1e-9) }
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
        }

        let allDocs = Array(documents.values)

        bm25Index = BM25Index(
            documents: allDocs,
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )

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

    public func search(
        query queryEmbedding: [Float],
        numResults: Int? = nil,
        threshold: Float? = nil
    ) async throws -> [VecturaSearchResult] {
        if queryEmbedding.count != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: queryEmbedding.count
            )
        }

        let norm = l2Norm(queryEmbedding)
        let normalizedQuery = queryEmbedding.map { $0 / (norm + 1e-9) }

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
        return results
    }

    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil,
        model: VecturaModelSource = .default
    ) async throws -> [VecturaSearchResult] {
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: model)
        }

        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(model)")
        }

        // Initialize BM25 index if needed
        if bm25Index == nil {
            let docs = documents.values.map { $0 }
            bm25Index = BM25Index(
                documents: docs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }

        // Get vector similarity results
        let queryEmbeddingTensor = try modelBundle.encode(query)
        let queryEmbeddingFloatArray = await tensorToArray(queryEmbeddingTensor)
        let vectorResults = try await search(
            query: queryEmbeddingFloatArray,
            numResults: nil,
            threshold: nil
        )

        let bm25Results =
        bm25Index?.search(
            query: query,
            topK: documents.count
        ) ?? []

        // Create a map of document IDs to their BM25 scores
        let bm25Scores = Dictionary(
            bm25Results.map { ($0.document.id, $0.score) },
            uniquingKeysWith: { first, _ in first }
        )

        // Combine scores using hybrid scoring
        var hybridResults = vectorResults.map { result in
            let bm25Score = bm25Scores[result.id] ?? 0
            let hybridScore = VecturaDocument(
                id: result.id,
                text: result.text,
                embedding: []
            ).hybridScore(
                vectorScore: result.score,
                bm25Score: bm25Score,
                weight: config.searchOptions.hybridWeight
            )

            return VecturaSearchResult(
                id: result.id,
                text: result.text,
                score: hybridScore,
                createdAt: result.createdAt
            )
        }

        hybridResults.sort { $0.score > $1.score }

        if let threshold = threshold ?? config.searchOptions.minThreshold {
            hybridResults = hybridResults.filter { $0.score >= threshold }
        }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(hybridResults.prefix(limit))
    }

    @_disfavoredOverload
    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws -> [VecturaSearchResult] {
        try await search(query: query, numResults: numResults, threshold: threshold, model: .id(modelId))
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

    public func deleteDocuments(ids: [UUID]) async throws {
        if bm25Index != nil {
            let remainingDocs = documents.values.filter { !ids.contains($0.id) }
            bm25Index = BM25Index(
                documents: Array(remainingDocs),
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }

        for id in ids {
            documents[id] = nil
            normalizedEmbeddings[id] = nil

            let documentURL = storageDirectory.appendingPathComponent("\(id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
    }

    public func updateDocument(
        id: UUID,
        newText: String,
        model: VecturaModelSource = .default
    ) async throws {
        try await deleteDocuments(ids: [id])

        _ = try await addDocument(text: newText, id: id, model: model)
    }

    @_disfavoredOverload
    public func updateDocument(
        id: UUID,
        newText: String,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws {
        try await updateDocument(id: id, newText: newText, model: .id(modelId))
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
                let normalized = doc.embedding.map { $0 / (norm + 1e-9) }
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

    private func tensorToArray(_ tensor: MLTensor) async -> [Float] {
        let shaped = await tensor.cast(to: Float.self).shapedArray(of: Float.self)
        return shaped.scalars
    }

    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(into: 0) { $0 += $1.0 * $1.1 }
    }

    private func l2Norm(_ v: [Float]) -> Float {
        sqrt(dotProduct(v, v))
    }
}

internal extension Bert {
    static func loadModelBundle(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
        switch source {
        case .id(let modelId):
            try await loadModelBundle(from: modelId)
        case .folder(let url):
            try await loadModelBundle(from: url)
        }
    }
}
