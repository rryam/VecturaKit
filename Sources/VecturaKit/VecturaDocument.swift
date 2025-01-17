import Foundation
import MLX

/// A document stored in the vector database.
public struct VecturaDocument: Identifiable, Codable {

    /// The unique identifier of the document.
    public let id: UUID
    
    /// The text content of the document.
    public let text: String
    
    /// The vector embedding of the document.
    public let embedding: MLXArray
    
    /// The timestamp when the document was created.
    public let createdAt: Date
    
    /// Creates a new document with the given properties.
    ///
    /// - Parameters:
    ///   - id: The unique identifier for the document. If nil, a new UUID will be generated.
    ///   - text: The text content of the document.
    ///   - embedding: The vector embedding of the document.
    public init(id: UUID? = nil, text: String, embedding: MLXArray) {
        self.id = id ?? UUID()
        self.text = text
        self.embedding = embedding
        self.createdAt = Date()
    }

    enum CodingKeys: String, CodingKey {
        case id
        case text
        case embedding
        case createdAt
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(text, forKey: .text)
        try container.encode(embedding.asArray(Float.self), forKey: .embedding)
        try container.encode(createdAt, forKey: .createdAt)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(UUID.self, forKey: .id)
        self.text = try container.decode(String.self, forKey: .text)
        let array = try container.decode([Float].self, forKey: .embedding)
        self.embedding = MLXArray(array)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
    }
}
