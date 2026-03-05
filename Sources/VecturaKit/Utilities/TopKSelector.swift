import Foundation

/// Keeps only the top K elements using a bounded min-heap.
///
/// The heap root stores the lowest-ranked element currently retained, so inserts
/// are `O(log k)` and the full pass is `O(n log k)`.
struct TopKSelector<Element> {
  private let maxCount: Int
  private let isHigherRanked: (Element, Element) -> Bool
  private var heap: [Element] = []

  init(
    maxCount: Int,
    isHigherRanked: @escaping (Element, Element) -> Bool
  ) {
    precondition(maxCount > 0, "maxCount must be greater than zero")
    self.maxCount = maxCount
    self.isHigherRanked = isHigherRanked
    self.heap.reserveCapacity(maxCount)
  }

  mutating func insert(_ element: Element) {
    if heap.count < maxCount {
      heap.append(element)
      siftUp(from: heap.count - 1)
      return
    }

    guard let currentLowest = heap.first, isHigherRanked(element, currentLowest) else {
      return
    }

    heap[0] = element
    siftDown(from: 0)
  }

  func sortedElements() -> [Element] {
    heap.sorted(by: isHigherRanked)
  }

  private func isLowerRanked(_ lhs: Element, than rhs: Element) -> Bool {
    isHigherRanked(rhs, lhs)
  }

  private mutating func siftUp(from index: Int) {
    var childIndex = index

    while childIndex > 0 {
      let parentIndex = (childIndex - 1) / 2
      guard isLowerRanked(heap[childIndex], than: heap[parentIndex]) else {
        break
      }

      heap.swapAt(childIndex, parentIndex)
      childIndex = parentIndex
    }
  }

  private mutating func siftDown(from index: Int) {
    var parentIndex = index

    while true {
      let leftChildIndex = 2 * parentIndex + 1
      let rightChildIndex = leftChildIndex + 1
      var candidateIndex = parentIndex

      if leftChildIndex < heap.count,
        isLowerRanked(heap[leftChildIndex], than: heap[candidateIndex]) {
        candidateIndex = leftChildIndex
      }

      if rightChildIndex < heap.count,
        isLowerRanked(heap[rightChildIndex], than: heap[candidateIndex]) {
        candidateIndex = rightChildIndex
      }

      guard candidateIndex != parentIndex else {
        return
      }

      heap.swapAt(parentIndex, candidateIndex)
      parentIndex = candidateIndex
    }
  }
}
