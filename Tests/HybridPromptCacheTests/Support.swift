import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// One token per word, and a ChatML template that renders a past reply the
/// way Qwen3.5's does: without the `<think>\n\n</think>\n\n` it was
/// generated after. That difference is the reason a cache can resume only
/// from the end of the last message and not from the end of the reply.
struct ChatTokenizer: MLXLMCommon.Tokenizer {
    static let imStart = 1, imEnd = 2, newline = 3, think = 4, endThink = 5
    static let special = ["<|im_start|>": 1, "<|im_end|>": 2, "\n": 3, "<think>": 4,
                          "</think>": 5, "system": 6, "user": 7, "assistant": 8]
    /// Words take ids clear of the specials and of the tiny model's image and
    /// video tokens (500...503).
    static let words = 16..<496

    /// What follows every prompt: the start of a reply with thinking off.
    static let generationPrompt = [imStart, 8, newline, think, newline, newline, endThink,
                                   newline, newline]

    /// A word's id: `w123` is 123, anything else a stable hash.
    static func id(of word: some StringProtocol) -> Int {
        if word.first == "w", let n = Int(word.dropFirst()), words.contains(n) { return n }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in word.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3 }
        return words.lowerBound + Int(hash % UInt64(words.count))
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated()
        {
            if index > 0 { ids.append(Self.newline) }
            ids += line.split(separator: " ").map { Self.id(of: $0) }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "w\($0)" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { Self.special[token] }
    func convertIdToToken(_ id: Int) -> String? { "w\(id)" }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var ids: [Int] = []
        for message in messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            ids += [Self.imStart, Self.special[role] ?? 7, Self.newline]
            ids += encode(text: content, addSpecialTokens: false)
            ids += [Self.imEnd, Self.newline]
        }
        return ids + Self.generationPrompt
    }
}

struct ChatProcessor: UserInputProcessor {
    let tokenizer = ChatTokenizer()

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        return LMInput(tokens: MLXArray(try tokenizer.applyChatTemplate(messages: messages)))
    }
}

/// The two tiny hybrids: half their layers attention, half gated delta.
enum Tiny {
    /// Qwen3.5 as the VLM factory builds it, which is how the 27B loads. It
    /// carries its rope position delta in `LMOutput.State`, and refuses to
    /// continue a warm cache without it.
    static func qwen35(seed: UInt64 = 1) throws -> any LanguageModel {
        let json = """
            {
                "model_type": "qwen3_5_vl",
                "image_token_id": 500, "video_token_id": 501,
                "vision_start_token_id": 502, "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                    "vocab_size": 512, "full_attention_interval": 2,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                    "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
                    "rope_parameters": {
                        "type": "default", "mrope_section": [8, 4, 4],
                        "rope_theta": 100000.0, "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 64
                }
            }
            """
        let config = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: seed)) { MLXVLM.Qwen35(config) }
    }

    /// Qwen3.5 as the LLM factory builds it: text only, no carried state.
    static func qwen35Text(seed: UInt64 = 1) throws -> any LanguageModel {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "vocab_size": 512,
                "full_attention_interval": 2
            }
            """
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self,
                                              from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: seed)) { Qwen35TextModel(config) }
    }

    static func context(_ model: any LanguageModel) -> ModelContext {
        ModelContext(configuration: ModelConfiguration(id: "tiny"), model: model,
                     processor: ChatProcessor(), tokenizer: ChatTokenizer())
    }
}

/// SplitMix64: a seeded generator, so a failing case can be run again.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}
