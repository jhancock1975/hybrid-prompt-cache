import XCTest
@testable import HybridPromptCache

/// Which part of a prompt the model has already read, decided without a
/// model: token ids in, positions out.
///
/// The model half - restoring the caches, reading what is new, rewinding -
/// is `ConversationTests`.
final class DecidingTests: XCTestCase {

    // Token ids that stand in for a ChatML chat: 9 is `<|im_end|>`, 8 is the
    // newline after it, everything else is text.
    private let end = 9, newline = 8

    // MARK: - Where a prompt can be checkpointed

    /// A system message, a user message, then the generation prompt: one
    /// checkpoint after each message, none inside the generation prompt.
    func testCheckpointsFallAfterTheFirstAndLastMessages() {
        //            system...  end nl  user... end nl  gen prompt
        let tokens = [1, 2, 3, 9, 8, 4, 5, 9, 8, 6, 7]
        XCTAssertEqual(PromptCache.checkpoints(in: tokens, endOfTurn: [end], newline: newline),
                       [5, 9])
    }

    /// Three messages: the middle one's end is no use to the next turn that
    /// the last one's end does not already cover, so it is not kept.
    func testOnlyTheFirstAndLastMessageEndsAreKept() {
        let tokens = [1, 9, 8, 2, 9, 8, 3, 9, 8, 6]
        XCTAssertEqual(PromptCache.checkpoints(in: tokens, endOfTurn: [end], newline: newline),
                       [3, 9])
    }

    func testOneMessageHasOneCheckpoint() {
        let tokens = [1, 2, 9, 8, 6, 7]
        XCTAssertEqual(PromptCache.checkpoints(in: tokens, endOfTurn: [end], newline: newline),
                       [4])
    }

    /// A template with no newline after its turns still has boundaries.
    func testANewlineIsOptional() {
        let tokens = [1, 9, 2, 9, 6]
        XCTAssertEqual(PromptCache.checkpoints(in: tokens, endOfTurn: [end], newline: nil),
                       [2, 4])
    }

    /// **Something must be left to generate from.** A prompt that ends on a
    /// message boundary has no generation prompt, and the model cannot be
    /// started on zero tokens.
    func testACheckpointAtTheVeryEndIsNotOne() {
        let tokens = [1, 2, 9, 8]
        XCTAssertEqual(PromptCache.checkpoints(in: tokens, endOfTurn: [end], newline: newline),
                       [])
    }

    func testAPromptWithNoTurnsHasNoCheckpoints() {
        XCTAssertEqual(PromptCache.checkpoints(in: [1, 2, 3], endOfTurn: [end], newline: newline),
                       [])
    }

    // MARK: - Which stored prefix a prompt can use

    /// The next turn of the same conversation: everything the last turn read
    /// up to its final checkpoint is a prefix of this prompt.
    func testTheNextTurnResumesFromTheLastTurnsFinalCheckpoint() {
        let stored = [Stored(tokens: [1, 2, 9, 8, 3, 9, 8], checkpoints: [4, 7])]
        let next = [1, 2, 9, 8, 3, 9, 8, 5, 9, 8, 4, 9, 8, 6]
        let choice = PromptCache.choose(for: next, among: stored)
        XCTAssertEqual(choice?.entry, 0)
        XCTAssertEqual(choice?.position, 7)
    }

    /// Another conversation with the same system prompt: only the system
    /// message is shared, and that is the checkpoint used.
    func testAnotherConversationResumesFromTheSystemPrompt() {
        let stored = [Stored(tokens: [1, 2, 9, 8, 3, 9, 8], checkpoints: [4, 7])]
        let other = [1, 2, 9, 8, 7, 7, 9, 8, 6]
        XCTAssertEqual(PromptCache.choose(for: other, among: stored)?.position, 4)
    }

    /// **The system prompt changed**, so nothing stored is a prefix and the
    /// prompt is read from the start.
    func testAChangedSystemPromptUsesNothing() {
        let stored = [Stored(tokens: [1, 2, 9, 8, 3, 9, 8], checkpoints: [4, 7])]
        XCTAssertNil(PromptCache.choose(for: [1, 5, 9, 8, 3, 9, 8, 6], among: stored))
    }

    /// The deepest usable checkpoint across every stored prefix wins.
    func testTheLongestUsablePrefixWins() {
        let stored = [
            Stored(tokens: [1, 2, 9, 8, 3, 9, 8], checkpoints: [4, 7]),
            Stored(tokens: [1, 2, 9, 8, 3, 9, 8, 5, 9, 8], checkpoints: [4, 10]),
        ]
        let next = [1, 2, 9, 8, 3, 9, 8, 5, 9, 8, 6]
        let choice = PromptCache.choose(for: next, among: stored)
        XCTAssertEqual(choice?.entry, 1)
        XCTAssertEqual(choice?.position, 10)
    }

    /// **Only at a checkpoint.** Two prompts can share tokens past the
    /// system message without sharing a checkpoint there; the recurrent
    /// layers cannot be rewound to an arbitrary token, only restored to a
    /// state that was saved.
    func testASharedPrefixWithNoCheckpointInsideItIsNotUsed() {
        let stored = [Stored(tokens: [1, 2, 9, 8, 3, 3, 3, 9, 8], checkpoints: [4, 9])]
        let next = [1, 2, 9, 8, 3, 3, 3, 4, 9, 8, 6]
        XCTAssertEqual(PromptCache.choose(for: next, among: stored)?.position, 4)
    }

    /// A stored prefix as long as the whole prompt leaves nothing to start
    /// generating from, so it is not a candidate.
    func testAPrefixCoveringTheWholePromptIsNotUsed() {
        let stored = [Stored(tokens: [1, 9, 8, 6], checkpoints: [3, 4])]
        XCTAssertEqual(PromptCache.choose(for: [1, 9, 8, 6], among: stored)?.position, 3)
    }

    // MARK: - What is kept

    func testTheOldestPrefixGoesWhenAThirdArrives() {
        var kept: [Int] = []
        for id in 1...3 { kept = PromptCache.keeping(kept, adding: id, limit: 2) }
        XCTAssertEqual(kept, [3, 2])
    }

    func testReturningAPrefixMakesItTheNewest() {
        XCTAssertEqual(PromptCache.keeping([2, 1], adding: 1, limit: 2), [1, 2])
    }

    // MARK: - Lending and keeping, with entries that hold no layers

    /// An entry with no layers is a real one with nothing to put back, so
    /// the whole lend-and-keep cycle runs here without a model.
    private func entry(_ tokens: [Int], at checkpoints: [Int]) -> PromptCache.Entry {
        let entry = PromptCache.Entry(cache: [])
        for position in checkpoints { entry.checkpoint(at: position, of: tokens, state: nil) }
        return entry
    }

    /// A prompt borrows the prefix it resumes from, so two prompts cannot
    /// write into one cache; it is there again once kept.
    func testALentPrefixIsOutUntilItIsKept() async {
        let prompts = PromptCache()
        let turn = [1, 9, 8, 2, 9, 8, 6]
        await prompts.keep(entry(turn, at: [3, 6]))
        let lent = await prompts.lend(for: turn + [9, 8, 3, 9, 8, 6])
        XCTAssertEqual(lent?.position, 6)
        let again = await prompts.lend(for: turn + [9, 8, 3, 9, 8, 6])
        XCTAssertNil(again, "one prefix was lent to two prompts")
        if let lent { await prompts.keep(lent.entry) }
        let held = await prompts.held
        XCTAssertEqual(held, [6])
    }

    /// Put back to a shallower checkpoint, an entry forgets the deeper one:
    /// it described tokens the new prompt does not have.
    func testRestoringForgetsDeeperCheckpoints() async {
        let prompts = PromptCache()
        await prompts.keep(entry([1, 9, 8, 2, 9, 8, 6], at: [3, 6]))
        let lent = await prompts.lend(for: [1, 9, 8, 7, 9, 8, 6])
        XCTAssertEqual(lent?.position, 3)
        XCTAssertEqual(lent?.entry.checkpoints, [3])
        XCTAssertEqual(lent?.entry.tokens, [1, 9, 8])
    }

    /// **A prefix past the cap is not held.** The conversation still works;
    /// it is read afresh each turn.
    func testAPrefixLongerThanTheCapIsNotKept() async {
        let prompts = PromptCache(longest: 5)
        await prompts.keep(entry([1, 9, 8, 2, 2, 2, 9, 8, 6], at: [3, 8]))
        let held = await prompts.held
        XCTAssertEqual(held, [])
    }

    func testClearingForgetsEverything() async {
        let prompts = PromptCache()
        await prompts.keep(entry([1, 9, 8, 2, 9, 8, 6], at: [3, 6]))
        await prompts.clear()
        let lent = await prompts.lend(for: [1, 9, 8, 2, 9, 8, 6, 6])
        XCTAssertNil(lent)
    }
}

/// A stored prefix as the chooser sees it.
struct Stored: PromptCache.Prefix {
    let tokens: [Int]
    let checkpoints: [Int]
}
