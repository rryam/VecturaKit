import Foundation
import OSLog

enum OpenAICompatibleRateLimitRetry {
    private static let maxRetryDelayNanoseconds: UInt64 = 30_000_000_000
    private static let minimumRetryAttempts = 0
    private static let maximumRetryAttempts = 6
    private static let minimumRetryBaseDelaySeconds: TimeInterval = 1
    private static let maximumRetryBaseDelaySeconds: TimeInterval = 30

    private static let retryAfterDateFormatters: [DateFormatter] = {
        let formats = [
            "EEE',' dd MMM yyyy HH':'mm':'ss zzz",
            "EEEE',' dd-MMM-yy HH':'mm':'ss zzz",
            "EEE MMM d HH':'mm':'ss yyyy",
        ]

        return formats.map { format in
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = format
            return formatter
        }
    }()

    static func data(
        for request: URLRequest,
        operation: String,
        logger: Logger,
        retryAttempts: Int,
        retryBaseDelaySeconds: TimeInterval,
        session: URLSession
    ) async throws -> (Data, URLResponse) {
        let maxRetries = min(max(retryAttempts, minimumRetryAttempts), maximumRetryAttempts)

        for attempt in 0...maxRetries {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                return (data, response)
            }

            guard httpResponse.statusCode == 429, attempt < maxRetries else {
                return (data, response)
            }

            let retryNumber = attempt + 1
            let delayNanoseconds = retryDelayNanoseconds(
                retryNumber: retryNumber,
                retryAfterHeader: httpResponse.value(forHTTPHeaderField: "Retry-After"),
                retryBaseDelaySeconds: retryBaseDelaySeconds
            )

            logger.warning(
                "OpenAI-compatible \(operation, privacy: .public) hit rate limit (retry \(retryNumber)/\(maxRetries)); retrying in \(formattedDelay(delayNanoseconds), privacy: .public)"
            )
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }

        return try await session.data(for: request)
    }

    private static func retryDelayNanoseconds(
        retryNumber: Int,
        retryAfterHeader: String?,
        retryBaseDelaySeconds: TimeInterval
    ) -> UInt64 {
        if let headerDelay = retryAfterDelayNanoseconds(from: retryAfterHeader) {
            return headerDelay
        }

        let exponent = min(max(retryNumber - 1, 0), 4)
        let multiplier = UInt64(1 << exponent)
        let baseDelayNanoseconds = nanoseconds(
            fromSeconds: min(
                max(retryBaseDelaySeconds, minimumRetryBaseDelaySeconds),
                maximumRetryBaseDelaySeconds
            )
        )
        return min(baseDelayNanoseconds * multiplier, maxRetryDelayNanoseconds)
    }

    private static func retryAfterDelayNanoseconds(from header: String?) -> UInt64? {
        guard let trimmedHeader = header?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmedHeader.isEmpty else {
            return nil
        }

        if let seconds = Double(trimmedHeader), seconds > 0 {
            return nanoseconds(fromSeconds: seconds)
        }

        for formatter in retryAfterDateFormatters {
            if let retryDate = formatter.date(from: trimmedHeader) {
                let seconds = retryDate.timeIntervalSinceNow
                guard seconds > 0 else {
                    return nil
                }
                return nanoseconds(fromSeconds: seconds)
            }
        }

        return nil
    }

    private static func nanoseconds(fromSeconds seconds: TimeInterval) -> UInt64 {
        let nanoseconds = seconds * 1_000_000_000
        let rounded = UInt64(max(0, nanoseconds.rounded()))
        return min(rounded, maxRetryDelayNanoseconds)
    }

    private static func formattedDelay(_ nanoseconds: UInt64) -> String {
        let seconds = Double(nanoseconds) / 1_000_000_000
        return String(format: "%.2fs", seconds)
    }
}
