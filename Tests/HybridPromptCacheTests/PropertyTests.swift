import XCTest
@testable import HybridPromptCache

/// The deciding half, over thousands of generated prompts rather than the
/// handful `DecidingTests` names. Every case is seeded; a failure prints the
/// seed that makes it again.
final class PropertyTests: XCTestCase {

    private let end = 9, newline = 8
    private let cases = 2_000

    /// A prompt from a small alphabet, heavy on turn ends so most have
    /// several messages, and sometimes a turn end with no newline after it.
    private func prompt(_ rng: inout SeededGenerator, length: ClosedRange<Int> = 0...30) -> [Int] {
        (0..<Int.random(in: length, using: &rng)).map { _ in
            switch Int.random(in: 0..<10, using: &rng) {
            case 0, 1: end
            case 2: newline
            default: Int.random(in: 1...5, using: &rng)
            }
        }
    }

    /// Where every message in `tokens` ends, before the end of the prompt.
    private func messageEnds(_ tokens: [Int], newline: Int?) -> [Int] {
        tokens.indices.filter { tokens[$0] == end }.map { index in
            index + 1 < tokens.count && tokens[index + 1] == newline ? index + 2 : index + 1
        }.filter { $0 < tokens.count }
    }

    // MARK: - checkpoints(in:)

    /// **At most two, both real message ends, and the outermost ones.** The
    /// first is the earliest end in the prompt and the last the latest, and
    /// neither is the prompt's very end.
    func testCheckpointsAreTheFirstAndLastMessageEnds() {
        for seed in 0..<UInt64(cases) {
            var rng = SeededGenerator(seed: seed)
            let tokens = prompt(&rng)
            let withNewline: Int? = Bool.random(using: &rng) ? newline : nil
            let marks = PromptCache.checkpoints(in: tokens, endOfTurn: [end],
                                                newline: withNewline)
            let ends = messageEnds(tokens, newline: withNewline)
            let context = "seed \(seed): \(tokens) newline \(String(describing: withNewline))"

            XCTAssertLessThanOrEqual(marks.count, 2, context)
            XCTAssertEqual(marks, marks.sorted(), context)
            XCTAssertEqual(Set(marks).count, marks.count, context)
            for mark in marks {
                XCTAssertTrue(mark > 0 && mark < tokens.count, "\(context) mark \(mark)")
                XCTAssertTrue(ends.contains(mark), "\(context): \(mark) is not a message end")
            }
            XCTAssertEqual(marks.first, ends.first, context)
            XCTAssertEqual(marks.last, ends.last, context)
        }
    }

    // MARK: - choose(for:among:)

    /// **The deepest shared checkpoint, found the slow way.** Every pair of
    /// stored prefix and checkpoint is tried; `choose` must land on the same
    /// depth, on a prefix that really matches there, with a token left over.
    func testChooseFindsTheDeepestMatchingCheckpoint() {
        for seed in 0..<UInt64(cases) {
            var rng = SeededGenerator(seed: seed)
            // Stored prefixes that often share a beginning with the prompt,
            // as turns of one conversation do.
            let base = prompt(&rng, length: 0...20)
            let stored: [Stored] = (0..<Int.random(in: 0...4, using: &rng)).map { _ in
                let tokens = Array(base.prefix(Int.random(in: 0...base.count, using: &rng)))
                    + prompt(&rng, length: 0...10)
                let checkpoints = Set((0..<Int.random(in: 0...3, using: &rng)).map { _ in
                    Int.random(in: 1...max(1, tokens.count), using: &rng)
                }).sorted()
                return Stored(tokens: tokens, checkpoints: checkpoints)
            }
            let tokens = Array(base.prefix(Int.random(in: 0...base.count, using: &rng)))
                + prompt(&rng, length: 0...10)

            var deepest = 0
            for prefix in stored {
                for position in prefix.checkpoints
                where position < tokens.count && position <= prefix.tokens.count
                    && Array(tokens[..<position]) == Array(prefix.tokens[..<position])
                {
                    deepest = max(deepest, position)
                }
            }
            let choice = PromptCache.choose(for: tokens, among: stored)
            let context = "seed \(seed): prompt \(tokens) among \(stored)"

            guard let choice else {
                XCTAssertEqual(deepest, 0, context)
                continue
            }
            XCTAssertEqual(choice.position, deepest, context)
            let chosen = stored[choice.entry]
            XCTAssertTrue(chosen.checkpoints.contains(choice.position), context)
            XCTAssertLessThan(choice.position, tokens.count, context)
            XCTAssertEqual(Array(tokens[..<choice.position]),
                           Array(chosen.tokens[..<choice.position]), context)
        }
    }

    // MARK: - keeping(_:adding:limit:)

    /// Newest first, no duplicates, never over the limit, and the rest in
    /// the order they were.
    func testKeepingIsANewestFirstListOfDistinctIds() {
        for seed in 0..<UInt64(cases) {
            var rng = SeededGenerator(seed: seed)
            let limit = Int.random(in: 1...4, using: &rng)
            var kept: [Int] = []
            for _ in 0..<Int.random(in: 1...12, using: &rng) {
                let id = Int.random(in: 0...5, using: &rng)
                let before = kept
                kept = PromptCache.keeping(kept, adding: id, limit: limit)
                let context = "seed \(seed): \(before) + \(id), limit \(limit)"

                XCTAssertEqual(kept.first, id, context)
                XCTAssertEqual(Set(kept).count, kept.count, context)
                XCTAssertEqual(kept.count, min(limit, Set(before + [id]).count), context)
                XCTAssertEqual(Array(kept.dropFirst()),
                               Array(before.filter { $0 != id }.prefix(limit - 1)), context)
            }
        }
    }

    // MARK: - The store, over a run of prompts

    /// **A conversation always picks up where its last prompt stopped.**
    /// Prompts run through the store as `resume` runs them - lend, read to
    /// each checkpoint, keep - on entries with no layers. After each one:
    /// the store is within its limits, and asking again with a prompt that
    /// extends this one lends back the end of its last message.
    func testAPromptResumesFromTheEndOfTheLastOne() async {
        for seed in 0..<UInt64(cases / 4) {
            var rng = SeededGenerator(seed: seed)
            let limit = Int.random(in: 1...3, using: &rng)
            let longest = Int.random(in: 4...40, using: &rng)
            let prompts = PromptCache(limit: limit, longest: longest)
            var history = prompt(&rng, length: 1...8) + [end, newline]

            for _ in 0..<Int.random(in: 1...6, using: &rng) {
                history += prompt(&rng, length: 1...6) + [end, newline]
                let tokens = history + [7]
                let marks = PromptCache.checkpoints(in: tokens, endOfTurn: [end],
                                                    newline: newline)
                let entry: PromptCache.Entry
                var read = 0
                if let lent = await prompts.lend(for: tokens) {
                    (entry, read) = lent
                } else {
                    entry = PromptCache.Entry(cache: [])
                }
                for mark in marks where mark > read {
                    entry.checkpoint(at: mark, of: tokens, state: nil)
                    read = mark
                }
                await prompts.keep(entry)

                let held = await prompts.held
                let context = "seed \(seed): \(tokens), limit \(limit), longest \(longest)"
                XCTAssertLessThanOrEqual(held.count, limit, context)
                XCTAssertTrue(held.allSatisfy { $0 <= longest }, context)

                let next = await prompts.lend(for: tokens + [7])
                if let last = marks.last, last <= longest {
                    XCTAssertEqual(next?.position, last, context)
                    XCTAssertEqual(next?.entry.tokens, Array(tokens.prefix(last)), context)
                } else {
                    XCTAssertTrue((next?.position ?? 0) <= longest, context)
                }
                if let next { await prompts.keep(next.entry) }
                history += [Int.random(in: 1...5, using: &rng), end, newline]
            }
        }
    }
}
