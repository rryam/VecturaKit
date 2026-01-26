import Foundation
import Testing
@testable import VecturaKit

@Suite("VecturaConfig Validation")
struct VecturaConfigValidationTests {
  private func expectInvalidConfig(
    _ makeConfig: () throws -> VecturaConfig,
    messageContains: String
  ) {
    do {
      _ = try makeConfig()
      Issue.record("Expected invalidInput error")
    } catch let error as VecturaError {
      switch error {
      case .invalidInput(let reason):
        #expect(reason.contains(messageContains))
      default:
        Issue.record("Unexpected error: \(error)")
      }
    } catch {
      Issue.record("Unexpected error type: \(error)")
    }
  }

  @Test("Rejects invalid database names")
  func rejectsInvalidDatabaseNames() {
    expectInvalidConfig(
      { try VecturaConfig(name: "   ") },
      messageContains: "Database name cannot be empty"
    )
    expectInvalidConfig(
      { try VecturaConfig(name: ".") },
      messageContains: "Database name cannot be '.'"
    )
    expectInvalidConfig(
      { try VecturaConfig(name: "db/name") },
      messageContains: "path separators"
    )
  }

  @Test("Rejects invalid search options")
  func rejectsInvalidSearchOptions() {
    expectInvalidConfig(
      { try VecturaConfig(name: "valid", searchOptions: .init(defaultNumResults: 0)) },
      messageContains: "defaultNumResults"
    )
    expectInvalidConfig(
      { try VecturaConfig(name: "valid", searchOptions: .init(minThreshold: 1.2)) },
      messageContains: "minThreshold"
    )
    expectInvalidConfig(
      { try VecturaConfig(name: "valid", searchOptions: .init(k1: 0)) },
      messageContains: "k1 must be greater than 0"
    )
    expectInvalidConfig(
      { try VecturaConfig(name: "valid", searchOptions: .init(b: -0.1)) },
      messageContains: "b must be between"
    )
  }
}
