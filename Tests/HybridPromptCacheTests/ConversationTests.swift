import MLX
import MLXLMCommon
import XCTest
@testable import HybridPromptCache

/// The model half, on two tiny random-weight Qwen3.5s, so it runs anywhere
/// in seconds and needs no download.
///
/// **What is claimed.**
/// 1. A cached conversation generates what an uncached one does, token for
///    token, through the library's own `generateTokens`.
/// 2. Each turn resumes from the end of the previous question, not from the
///    end of the previous reply - the template re-renders the reply, so a
///    cache holding it could never be extended.
/// 3. **Putting a cache back is exact**, however far it had moved on: two
///    stores shown the same prompts, one of whose caches ran further past
///    each checkpoint, give the same logits to the bit.
/// 4. A cached prompt agrees with a cold read of it as closely as upstream
///    holds its own warm continuations: within ten times the drift of reading
///    the same prompt token by token, or 1e-3.
///
/// Claims 3 and 4 are properties, over generated conversations of every
/// shape. Claim 4 is not claim 3 with a tolerance: reading a prompt in pieces
/// moves the floating point whether or not anything was cached - at a chunk
/// size of one, by 0.003 on these models - so only claim 3 can be exact.
///
/// Both models mix attention layers with gated-delta layers, so both halves
/// of `restore` run: cutting back and putting arrays back.
final class ConversationTests: XCTestCase {

    private static let models: [(name: String,
                                 make: @Sendable (UInt64) throws -> any LanguageModel)] = [
        ("VLM Qwen3.5", { try Tiny.qwen35(seed: $0) }),
        ("LLM Qwen3.5", { try Tiny.qwen35Text(seed: $0) }),
    ]

    private static let greedy = GenerateParameters(maxTokens: 8, temperature: 0)

    private func generate(_ input: LMInput, cache: [any KVCache]? = nil,
                          state: LMOutput.State? = nil,
                          context: ModelContext) async throws -> [Int] {
        var tokens: [Int] = []
        let stream = try generateTokens(input: input, cache: cache, state: state,
                                        parameters: Self.greedy, context: context)
        for await piece in stream {
            if let token = piece.token { tokens.append(token) }
        }
        return tokens
    }

    func testTheTinyModelsAreHybrids() throws {
        for (name, make) in Self.models {
            let cache = try make(1).newCache(parameters: nil)
            XCTAssertTrue(cache.contains { $0.isTrimmable }, "\(name) has no attention layer")
            XCTAssertTrue(cache.contains { !$0.isTrimmable }, "\(name) has no recurrent layer")
        }
    }

    /// Claims 1 and 2, over a four-turn conversation.
    func testAConversationGeneratesWhatItWouldHaveUncached() async throws {
        let questions = [
            "here are my notes : the red house has four windows and a green door",
            "which house has a green door",
            "how many windows does it have",
            "summarise the notes in one line",
        ]
        for (name, make) in Self.models {
            let context = Tiny.context(try make(1))
            let prompts = PromptCache()
            let boundaries = MessageBoundaries(tokenizer: context.tokenizer)
            var chat: [Chat.Message] = [.system("you answer questions about the notes")]
            var previous: (prompt: [Int], reply: [Int])?

            for (turn, question) in questions.enumerated() {
                chat.append(.user(question))
                let input = try await context.processor.prepare(input: UserInput(chat: chat))
                let tokens = input.text.tokens.asArray(Int.self)
                let cold = try await generate(input, context: context)

                let resumed = try await prompts.resume(input, model: context.model,
                                                       boundaries: boundaries)
                guard let resumed else { return XCTFail("\(name) turn \(turn): not resumed") }
                let warm = try await generate(resumed.rest, cache: resumed.entry.cache,
                                              state: resumed.state, context: context)
                await prompts.keep(resumed.entry)

                XCTAssertEqual(warm, cold, "\(name) turn \(turn)")
                if let previous {
                    XCTAssertFalse(tokens.starts(with: previous.prompt + previous.reply),
                                   "\(name) turn \(turn): the template kept the reply as generated, so this test proves nothing about re-rendering")
                    XCTAssertEqual(resumed.reused,
                                   previous.prompt.count - ChatTokenizer.generationPrompt.count,
                                   "\(name) turn \(turn): did not resume from the last question")
                } else {
                    XCTAssertEqual(resumed.reused, 0, "\(name): an empty store lent something")
                }
                chat.append(.assistant(context.tokenizer.decode(tokenIds: cold)))
                previous = (tokens, cold)
            }
        }
    }

    /// Another conversation with the same system prompt reads it once.
    func testAnotherConversationStartsFromTheSharedSystemPrompt() async throws {
        let system = "you answer questions about the notes"
        let systemTokens = 3 + ChatTokenizer().encode(text: system).count + 2
        for (name, make) in Self.models {
            let context = Tiny.context(try make(1))
            let prompts = PromptCache()
            let boundaries = MessageBoundaries(tokenizer: context.tokenizer)

            for (question, expected) in [("what colour is the door", 0),
                                         ("how many windows are there", systemTokens)] {
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [.system(system), .user(question)]))
                let cold = try await generate(input, context: context)
                let resumed = try await prompts.resume(input, model: context.model,
                                                       boundaries: boundaries)
                guard let resumed else { return XCTFail("\(name): not resumed") }
                let warm = try await generate(resumed.rest, cache: resumed.entry.cache,
                                              state: resumed.state, context: context)
                await prompts.keep(resumed.entry)
                XCTAssertEqual(warm, cold, "\(name): \(question)")
                XCTAssertEqual(resumed.reused, expected, "\(name): \(question)")
            }
        }
    }

    /// One message has no second checkpoint worth saving, so it is left to
    /// the ordinary path.
    func testOneMessageIsNotCached() async throws {
        let context = Tiny.context(try Tiny.qwen35())
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("is this an instruction or a question")]))
        let resumed = try await PromptCache().resume(
            input, model: context.model, boundaries: MessageBoundaries(tokenizer: context.tokenizer))
        XCTAssertNil(resumed)
    }

    // MARK: - Properties, over generated conversations

    /// A run of prompts as a chat app makes them: two conversations taking
    /// turns on one model, replies of any length, and now and then a changed
    /// system prompt, which nothing stored can serve.
    private func conversation(_ rng: inout SeededGenerator) -> [[Int]] {
        let tokenizer = ChatTokenizer()
        func words(_ range: ClosedRange<Int>) -> String {
            (0..<Int.random(in: range, using: &rng))
                .map { _ in "w\(Int.random(in: ChatTokenizer.words, using: &rng))" }
                .joined(separator: " ")
        }
        var system = words(1...12)
        var chats: [[[String: String]]] = [[], []]
        return (0..<Int.random(in: 2...7, using: &rng)).map { _ in
            if Int.random(in: 0..<6, using: &rng) == 0 { system = words(1...12) }
            let which = Int.random(in: 0...1, using: &rng)
            if !chats[which].isEmpty {
                chats[which].append(["role": "assistant", "content": words(0...10)])
            }
            chats[which].append(["role": "user", "content": words(1...20)])
            let messages = [["role": "system", "content": system]] + chats[which]
            return try! tokenizer.applyChatTemplate(messages: messages)
        }
    }

    private func lastLogits(_ output: LMOutput) -> MLXArray {
        output.logits[0..., -1, 0...]
    }

    private func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Claim 3. Two stores see the same prompts and so read them in the same
    /// pieces; after each prompt, each store's cache runs on past the end by
    /// a different random number of tokens - the reply being generated -
    /// before it is given back. Anything the restore failed to undo shows as
    /// a difference, and there must be none at all.
    func testRestoringIsExactHoweverFarTheCacheRanOn() async throws {
        var checked = 0
        for seed in 0..<UInt64(24) {
            var rng = SeededGenerator(seed: seed)
            for (name, make) in Self.models {
                let model = try make(seed)
                let limit = Int.random(in: 1...3, using: &rng)
                let step = [1, 3, 7, 16, 512].randomElement(using: &rng)!
                let stores = [PromptCache(limit: limit), PromptCache(limit: limit)]
                let boundaries = MessageBoundaries(tokenizer: ChatTokenizer())

                for (turn, tokens) in conversation(&rng).enumerated() {
                    let input = LMInput(tokens: MLXArray(tokens))
                    let context = "seed \(seed) \(name) turn \(turn) step \(step): \(tokens)"
                    var logits: [MLXArray] = []
                    var reused: [Int] = []
                    for store in stores {
                        let resumed = try await store.resume(input, model: model,
                                                             boundaries: boundaries,
                                                             prefillStep: step)
                        guard let resumed else { XCTFail(context); break }
                        var output = model(resumed.rest.text[text: .newAxis],
                                           cache: resumed.entry.cache, state: resumed.state)
                        logits.append(lastLogits(output))
                        reused.append(resumed.reused)
                        for _ in 0..<Int.random(in: 0...12, using: &rng) {
                            let next = Int.random(in: ChatTokenizer.words, using: &rng)
                            output = model(LMInput.Text(tokens: MLXArray([Int32(next)])[.newAxis]),
                                           cache: resumed.entry.cache,
                                           state: output.state ?? resumed.state)
                        }
                        eval(resumed.entry.cache)
                        await store.keep(resumed.entry)
                    }
                    guard logits.count == 2 else { continue }
                    XCTAssertEqual(reused[0], reused[1], context)
                    XCTAssertEqual(difference(logits[0], logits[1]), 0, context)
                    checked += 1
                }
            }
        }
        print("prompts checked for an exact restore: \(checked)")
    }

    /// Claim 4, held to upstream's bar for a warm continuation
    /// (`ContinuationAssertions.assertWarmTextContinuation`): the drift of a
    /// token-by-token read from a one-pass read is the noise floor, and a
    /// cached prompt may drift ten times that, or 1e-3.
    func testACachedPromptAgreesWithAColdReadToTheNoiseFloor() async throws {
        var worst: (difference: Float, floor: Float) = (0, 0)
        for seed in 0..<UInt64(24) {
            var rng = SeededGenerator(seed: seed)
            for (name, make) in Self.models {
                let model = try make(seed)
                let prompts = PromptCache(limit: Int.random(in: 1...3, using: &rng))
                let boundaries = MessageBoundaries(tokenizer: ChatTokenizer())
                let step = [1, 3, 7, 16, 512].randomElement(using: &rng)!

                for (turn, tokens) in conversation(&rng).enumerated() {
                    let input = LMInput(tokens: MLXArray(tokens))
                    let context = "seed \(seed) \(name) turn \(turn) step \(step): \(tokens)"

                    let cold = lastLogits(model(input.text[text: .newAxis],
                                                cache: try model.newCache(parameters: nil),
                                                state: nil))
                    let byToken = try model.newCache(parameters: nil)
                    let state = PromptCache.prefill(tokens.dropLast()[...], into: byToken,
                                                    state: nil, model: model, step: 1)
                    let control = lastLogits(model(
                        LMInput.Text(tokens: MLXArray([Int32(tokens.last!)])[.newAxis]),
                        cache: byToken, state: state))
                    let floor = difference(control, cold)

                    let resumed = try await prompts.resume(input, model: model,
                                                           boundaries: boundaries,
                                                           prefillStep: step)
                    guard let resumed else { XCTFail(context); continue }
                    let warm = lastLogits(model(resumed.rest.text[text: .newAxis],
                                                cache: resumed.entry.cache,
                                                state: resumed.state))
                    await prompts.keep(resumed.entry)

                    let drift = difference(warm, cold)
                    if drift > worst.difference { worst = (drift, floor) }
                    XCTAssertLessThanOrEqual(drift, max(floor * 10, 1e-3),
                                             "\(context) (noise floor \(floor))")
                }
            }
        }
        print("largest drift from a cold read: \(worst.difference), noise floor there \(worst.floor)")
    }
}
