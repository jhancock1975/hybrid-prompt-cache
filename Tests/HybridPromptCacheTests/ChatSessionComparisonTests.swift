import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest
@testable import HybridPromptCache

/// Why this exists, counted: how many prompt tokens each turn reads.
///
/// mlx-swift-lm's `ChatSession` keeps a cache across turns. When the next
/// prompt does not extend it, it cuts the cache back to the common prefix,
/// but only if every layer can be cut back; otherwise it reads everything
/// again. With a template that re-renders the reply, the next prompt never
/// extends the cache, so on a hybrid every turn is read in full.
///
/// Counted three ways over the same conversation:
/// - `ChatSession` on the tiny Qwen3.5 hybrid: every turn reads the whole
///   prompt;
/// - `ChatSession` on a tiny pure-attention Qwen3, the control: it cuts back
///   and reads only what follows the common prefix;
/// - this package on the hybrid: it resumes from the end of the last
///   question and reads the reply, the new question and the generation
///   prompt.
///
/// **If upstream learns to resume hybrids, the first assertion fails**, and
/// that is the signal this package has done its job.
final class ChatSessionComparisonTests: XCTestCase {

    private let system = "you answer questions about the notes"
    private let questions = [
        "here are my notes : the red house has four windows and a green door",
        "which house has a green door",
        "how many windows does it have",
    ]

    /// Tiny Qwen3: attention only, so every layer can be cut back.
    private func attentionOnly() async throws -> any LanguageModel {
        let json = """
            {
                "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
                "vocab_size": 512
            }
            """
        return try await withRandomState(MLXRandom.RandomState(seed: 1)) {
            try await LLMTypeRegistry.shared.createModel(configuration: Data(json.utf8),
                                                         modelType: "qwen3")
        }
    }

    /// Tokens read and whole prompt length, per turn, through `ChatSession`.
    private func chatSession(_ model: any LanguageModel) async throws -> [(read: Int, whole: Int)] {
        let processor = RecordingProcessor()
        let context = ModelContext(configuration: ModelConfiguration(id: "tiny"), model: model,
                                   processor: processor, tokenizer: ChatTokenizer())
        let session = ChatSession(context, instructions: system,
                                  generateParameters: GenerateParameters(maxTokens: 8,
                                                                         temperature: 0))
        var turns: [(read: Int, whole: Int)] = []
        for question in questions {
            for try await piece in session.streamDetails(to: question) {
                if case .info(let info) = piece {
                    turns.append((info.promptTokenCount, processor.lastLength))
                }
            }
        }
        return turns
    }

    /// The same, through this package.
    private func hybridPromptCache(_ model: any LanguageModel) async throws -> [(read: Int, whole: Int)] {
        let context = Tiny.context(model)
        let prompts = PromptCache()
        let boundaries = MessageBoundaries(tokenizer: context.tokenizer)
        var chat: [Chat.Message] = [.system(system)]
        var turns: [(read: Int, whole: Int)] = []
        for question in questions {
            chat.append(.user(question))
            let input = try await context.processor.prepare(input: UserInput(chat: chat))
            let whole = input.text.tokens.size
            guard let resumed = try await prompts.resume(input, model: model,
                                                         boundaries: boundaries)
            else { throw XCTSkip("not resumed") }
            var reply: [Int] = []
            let stream = try generateTokens(
                input: resumed.rest, cache: resumed.entry.cache, state: resumed.state,
                parameters: GenerateParameters(maxTokens: 8, temperature: 0), context: context)
            for await piece in stream {
                if let token = piece.token { reply.append(token) }
            }
            await prompts.keep(resumed.entry)
            turns.append((whole - resumed.reused, whole))
            chat.append(.assistant(context.tokenizer.decode(tokenIds: reply)))
        }
        return turns
    }

    func testOnAHybridOnlyThisResumesAfterTheFirstTurn() async throws {
        let session = try await chatSession(try Tiny.qwen35())
        let control = try await chatSession(try await attentionOnly())
        let cached = try await hybridPromptCache(try Tiny.qwen35())
        for (name, turns) in [("ChatSession, hybrid", session),
                              ("ChatSession, attention only", control),
                              ("HybridPromptCache, hybrid", cached)] {
            print("\(name): " + turns.map { "\($0.read) of \($0.whole)" }.joined(separator: ", "))
        }
        XCTAssertEqual(session.count, questions.count)
        XCTAssertEqual(control.count, questions.count)

        for turn in 1..<questions.count {
            XCTAssertEqual(session[turn].read, session[turn].whole,
                           "ChatSession resumed a hybrid on turn \(turn + 1)")
            XCTAssertLessThan(control[turn].read, control[turn].whole,
                              "the control did not resume, so the comparison shows nothing")
            XCTAssertLessThan(cached[turn].read, cached[turn].whole)
            XCTAssertLessThan(cached[turn].read, session[turn].read)
        }
    }
}

/// `ChatProcessor`, noting how long each prompt it prepares is.
final class RecordingProcessor: UserInputProcessor, @unchecked Sendable {
    private let lock = NSLock()
    private var length = 0
    var lastLength: Int { lock.withLock { length } }

    func prepare(input: UserInput) async throws -> LMInput {
        let prepared = try await ChatProcessor().prepare(input: input)
        lock.withLock { length = prepared.text.tokens.size }
        return prepared
    }
}
