import Accelerate
import CoreML
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
/// A vector database implementation that stores and searches documents using their vector embeddings.
public class VecturaKit: VecturaProtocol {

    /// The configuration for this vector database instance.
    private let config: VecturaConfig

    /// In-memory cache of all documents.
    private var documents: [UUID: VecturaDocument]

    /// The storage directory for documents.
    private let storageDirectory: URL

    /// The storage provider that handles document persistence.
    private let storageProvider: VecturaStorage

    /// Cached normalized embeddings for faster searches.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]

    /// BM25 index for text search
    private var bm25Index: BM25Index?

    /// Swift-Embeddings model bundle that you can reuse (e.g. BERT, XLM-R, CLIP, etc.)
    private var bertModel: Bert.ModelBundle?

    // MARK: - Initialization

    public init(config: VecturaConfig) async throws {
        self.config = config
        self.documents = [:]

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

        // Instantiate the storage provider (currently the file-based implementation).
        self.storageProvider = try FileStorageProvider(storageDirectory: storageDirectory)

        // Load existing documents using the storage provider.
        let storedDocuments = try await storageProvider.loadDocuments()
        for doc in storedDocuments {
            self.documents[doc.id] = doc
            // Compute normalized embedding and store in cache.
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            self.normalizedEmbeddings[doc.id] = normalized
        }
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
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
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

        // Normalize the query vector
        let norm = l2Norm(queryEmbedding)
        var divisor = norm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(queryEmbedding, 1, &divisor, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))

        // Build a matrix of normalized document embeddings in row-major order
        var docIds = [UUID]()
        var matrix = [Float]()
        matrix.reserveCapacity(documents.count * config.dimension)  // Pre-allocate for better performance

        for doc in documents.values {
            if let normalized = normalizedEmbeddings[doc.id] {
                docIds.append(doc.id)
                matrix.append(contentsOf: normalized)
            }
        }

        let docsCount = docIds.count
        if docsCount == 0 {
            return []
        }

        let M = Int32(docsCount)  // number of rows (documents)
        let N = Int32(config.dimension)  // number of columns (embedding dimension)
        var similarities = [Float](repeating: 0, count: docsCount)

        // Convert Int32 to Int for LAPACK compatibility
        let mInt = Int(M)  // Convert number of rows
        let nInt = Int(N)  // Convert number of columns
        let ldInt = Int(N) // Convert leading dimension

        // Compute all similarities at once using matrix-vector multiplication
        // Matrix is in row-major order, so we use CblasNoTrans
        cblas_sgemv(
            CblasRowMajor,    // matrix layout
            CblasNoTrans,     // no transpose needed for row-major
            mInt,             // number of rows (documents) as Int
            nInt,             // number of columns (dimension) as Int
            1.0,              // alpha scaling factor
            matrix,           // matrix
            ldInt,            // leading dimension as Int
            normalizedQuery,  // vector
            1,                // vector increment
            0.0,              // beta scaling factor
            &similarities,    // result vector
            1                 // result increment
        )

        // Construct the results
        var results = [VecturaSearchResult]()
        results.reserveCapacity(docsCount)  // Pre-allocate for better performance

        for (i, similarity) in similarities.enumerated() {
            if let minT = threshold ?? config.searchOptions.minThreshold, similarity < minT {
                continue
            }
            if let doc = documents[docIds[i]] {
                results.append(
                    VecturaSearchResult(
                        id: doc.id,
                        text: doc.text,
                        score: similarity,
                        createdAt: doc.createdAt
                    )
                )
            }
        }

        results.sort { $0.score > $1.score }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
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
        try await search(
            query: query, numResults: numResults, threshold: threshold, model: .id(modelId))
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

    // MARK: - Public Properties
    
    /// Returns the number of documents currently stored in the vector database.
    public var documentCount: Int {
        return documents.count
    }
    
    /// Returns all documents currently stored in the vector database.
    /// - Returns: Array of VecturaDocument objects
    public func getAllDocuments() -> [VecturaDocument] {
        return Array(documents.values)
    }

    // MARK: - Private

    private func tensorToArray(_ tensor: MLTensor) async -> [Float] {
        let shaped = await tensor.cast(to: Float.self).shapedArray(of: Float.self)
        return shaped.scalars
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

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    static func loadModelBundle(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
        switch source {
        case .id(let modelId):
            try await loadModelBundle(from: modelId)
        case .folder(let url):
            try await loadModelBundle(from: url)
        }
    }
}
