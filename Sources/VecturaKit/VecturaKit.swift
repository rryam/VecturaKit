import Foundation
import Embeddings
import CoreML

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

    /// Swift-Embeddings model bundle that you can reuse (e.g. BERT, XLM-R, CLIP, etc.)
    /// In a real scenario, you might have your own logic or accept a user-provided model.
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

    // MARK: - Public

    /// Adds multiple documents to the vector store in batch.
    public func addDocuments(
        texts: [String],
        ids: [UUID]? = nil,
        modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    ) async throws -> [UUID] {
        // 1. Check ID count (if provided)
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }

        // 2. Load or reuse the embeddings model
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: modelId)
        }
        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(modelId)")
        }

        // 3. Batch-encode embeddings using swift-embeddings
        let embeddingsTensor = try modelBundle.batchEncode(texts)
        let shape = embeddingsTensor.shape  // e.g. [batchSize, hiddenSize]
        if shape.count != 2 {
            throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
        }
        // shape[0] == texts.count
        // shape[1] == config.dimension, presumably
        if shape[1] != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: shape[1]
            )
        }

        // 4. Convert MLTensor → Swift [Float]
        let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(of: Float.self)
        // embeddingShapedArray has shape [texts.count, dimension]
        let allScalars = embeddingShapedArray.scalars

        // 5. Create VecturaDocuments
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

        // 6. Normalize + store in memory
        for doc in documentsToSave {
            let norm = l2Norm(doc.embedding)
            let normalized = doc.embedding.map { $0 / (norm + 1e-9) }
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
        }

        // 7. Persist to disk in parallel
        try await withThrowingTaskGroup(of: Void.self) { group in
            // Copy the directory URL into a local constant:
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

    /// Adds a single document to the vector store.
    public func addDocument(
        text: String,
        id: UUID? = nil,
        modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    ) async throws -> UUID {
        let ids = try await addDocuments(
            texts: [text],
            ids: id.map { [$0] },
            modelId: modelId
        )
        return ids[0]
    }

    /// Searches for similar documents using a query text. (Text-based version)
    /// You could also provide a search(queryEmbedding:) variant if you have your own embedding.
    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil,
        modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    ) async throws -> [VecturaSearchResult] {
        // Embed the query
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: modelId)
        }
        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(modelId)")
        }
        let queryEmbeddingTensor = try modelBundle.encode(query)
        let queryEmbeddingFloatArray = await tensorToArray(queryEmbeddingTensor)

        return try await search(query: queryEmbeddingFloatArray, numResults: numResults, threshold: threshold)
    }

    /// Searches for similar documents given a pre-computed query embedding.
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

        // Normalize query embedding
        let norm = l2Norm(queryEmbedding)
        let normalizedQuery = queryEmbedding.map { $0 / (norm + 1e-9) }

        var results: [VecturaSearchResult] = []

        for doc in documents.values {
            guard let normDoc = normalizedEmbeddings[doc.id] else { continue }
            let similarity = dotProduct(normalizedQuery, normDoc)
            // Filter out below threshold
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

    /// Removes all documents from the store.
    public func reset() async throws {
        documents.removeAll()
        normalizedEmbeddings.removeAll()

        // Remove all JSON files
        let files = try FileManager.default.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil)
        for fileURL in files {
            try FileManager.default.removeItem(at: fileURL)
        }
    }

    /// Deletes multiple documents from the store.
    public func deleteDocuments(ids: [UUID]) async throws {
        for id in ids {
            documents[id] = nil
            normalizedEmbeddings[id] = nil
            
            let documentURL = storageDirectory.appendingPathComponent("\(id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
    }

    /// Updates a document in the store.
    public func updateDocument(
        id: UUID,
        newText: String,
        modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    ) async throws {
        // Delete old document
        try await deleteDocuments(ids: [id])
        
        // Add updated document with same ID
        _ = try await addDocument(text: newText, id: id, modelId: modelId)
    }

    // MARK: - Private

    private func loadDocuments() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil)

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
                loadErrors.append("Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
        }
    }

    /// Utility to convert a 1D MLTensor → [Float].
    private func tensorToArray(_ tensor: MLTensor) async -> [Float] {
        let shaped = await tensor.cast(to: Float.self).shapedArray(of: Float.self)
        return shaped.scalars
    }

    /// Dot product of two vectors.
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(into: 0) { $0 += $1.0 * $1.1 }
    }

    /// L2 norm of a vector.
    private func l2Norm(_ v: [Float]) -> Float {
        sqrt(dotProduct(v, v))
    }
}
