import Foundation

public enum OpenAICompatibleEmbedderError: LocalizedError {
    case invalidURL
    case invalidInput(String)
    case httpError(statusCode: Int, message: String)
    case vectorCountMismatch(expected: Int, received: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid embedding endpoint URL."
        case .invalidInput(let message):
            return message
        case .httpError(let statusCode, let message):
            return "Embedding request failed (\(statusCode)): \(message)"
        case .vectorCountMismatch(let expected, let received):
            return "Expected \(expected) embedding vectors, received \(received)."
        }
    }
}
