# Repository Guidelines

## Project Structure & Module Organization
VecturaKit ships as a Swift package with libraries `VecturaKit` (core storage + hybrid search) and `VecturaNLKit` (NaturalLanguage embeddings), plus CLI target `VecturaCLI`. The CLI entry point sits in `Sources/VecturaCLI`; sample executables live in `Sources/TestExamples` and `Sources/TestNLExamples`. Tests live under `Tests/VecturaKitTests` and `Tests/VecturaNLKitTests`. MLX support has been extracted to a separate repository at [rryam/VecturaMLXKit](https://github.com/rryam/VecturaMLXKit).

## Build, Test, and Development Commands
- `swift build` compiles libraries and executables (add `-c release` for performance validation).
- `swift run vectura-cli mock --db-name demo-db` seeds and exercises the default engine.
- `swift test` runs the Swift Testing suites; add `--filter SuiteName/TestName` to narrow scope.
- `swift package update` refreshes dependency pins before release branches or large upgrades.

## Coding Style & Naming Conventions
Follow Swift 6 defaults: two-space indentation, trailing commas in multi-line literals, and a 120-character soft wrap. Keep types UpperCamelCase, members lowerCamelCase, and CLI command enums verb-based (`case add`, `case search`). Use the existing async/await APIs, isolate file IO in helpers, and document public entry points with concise `///` comments.

## Testing Guidelines
Declare new Swift Testing suites with `@Suite` and `@Test` annotations, keeping test functions `async throws` and cleaning up resources explicitly within each test. Prefer per-test temporary directories over `~/Documents/VecturaKit`. Do not introduce new `XCTestCase` subclasses—Swift Testing is the required framework going forward. Update `codemagic.yaml` whenever the CI matrix or steps change.

## Commit & Pull Request Guidelines
Commits stay imperative and scoped (`Add cosine similarity guard`, `Adjust CLI mock seed`). Keep dependency bumps isolated with matching `Package.resolved` updates. Pull requests should outline intent, summarize functional impact, and describe verification steps—attach CLI output when behavior changes. Link issues with `Fixes #<id>` and confirm CI before requesting review.

## Contribution Checklist
Before opening a PR (or completing an automated change):

- [ ] `swift build` and `swift test` succeed locally without warnings.
- [ ] Run `swift run vectura-cli mock --db-name qa-db` to ensure smoke coverage.
- [ ] Persistent storage defaults and dimension negotiation stay intact—verify hybrid search thresholds and expected `VecturaError.dimensionMismatch` behavior.
- [ ] Public API or CLI flag changes are reflected in `README.md`, `codemagic.yaml`, and this guide.
- [ ] Remove temporary database artifacts from `~/Documents/VecturaKit` so tests stay deterministic.
- [ ] No legacy instruction files are reintroduced; `AGENTS.md` supersedes earlier agent guidance.
