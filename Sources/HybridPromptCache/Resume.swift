import Foundation
import MLX
import MLXLMCommon

/// The tokens that end a message, which is where a prompt can be
/// checkpointed.
public struct MessageBoundaries: Sendable {
    public var endOfTurn: Set<Int>
    /// The token a template writes after `endOfTurn`, if any. ChatML writes
    /// `<|im_end|>\n`, and the next prompt repeats that newline too.
    public var newline: Int?

    public init(endOfTurn: Set<Int>, newline: Int?) {
        self.endOfTurn = endOfTurn
        self.newline = newline
    }

    /// The end-of-turn markers of ChatML (Qwen), Llama 3, Gemma and Phi,
    /// whichever this tokenizer has.
    public init(tokenizer: any Tokenizer) {
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false)
        self.init(
            endOfTurn: Set(["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>"]
                .compactMap { tokenizer.convertTokenToId($0) }),
            newline: newline.count == 1 ? newline[0] : nil)
    }
}

/// A prompt ready to generate from: the cache it borrowed, holding every
/// message, and what is left to read - the generation prompt.
///
/// Pass `rest`, `entry.cache` and `state` to `generateTokens`, and give
/// `entry` back to the store with `keep` once the generation has run to its
/// end.
public struct Resumed {
    public let entry: PromptCache.Entry
    public let rest: LMInput
    public let state: LMOutput.State?
    /// Tokens that were already in the cache and were not read again.
    public let reused: Int
}

extension PromptCache {

    /// Read what this prompt has that the store does not, stopping at each
    /// checkpoint to save the cache, and hand back the rest to generate
    /// from.
    ///
    /// Nil - generate the ordinary way - when:
    /// - **the prompt carries an image or video.** Their features are not
    ///   tokens, and a prefix is only comparable token by token;
    /// - **it has a padding mask**, which the rest would be read against at
    ///   the wrong positions;
    /// - **the model's cache has a layer that can be neither cut back nor
    ///   put back** (see `Entry.restores(_:)`);
    /// - **it is one message.** Its only checkpoint is after that message,
    ///   and nothing later resumes from it unless the same message comes
    ///   again. Measured on Qwen3.5-27B, saving one for a short classifier
    ///   prompt took a call from 0.20s to 0.36s for a checkpoint nothing
    ///   ever used.
    ///
    /// - Parameters:
    ///   - input: the prepared prompt, from the model's `UserInputProcessor`.
    ///   - model: the model the store belongs to.
    ///   - boundaries: where messages end in this model's template.
    ///   - prefillStep: tokens per model call while reading. The library's
    ///     own default.
    public nonisolated func resume(
        _ input: LMInput, model: any LanguageModel, boundaries: MessageBoundaries,
        prefillStep: Int = 512
    ) async throws -> Resumed? {
        guard input.image == nil, input.video == nil else { return nil }
        if let mask = input.text.mask, !mask.all().item(Bool.self) { return nil }
        let tokens = input.text.tokens.asArray(Int.self)
        let marks = Self.checkpoints(in: tokens, endOfTurn: boundaries.endOfTurn,
                                     newline: boundaries.newline)
        guard marks.count > 1 else { return nil }

        let entry: PromptCache.Entry
        var read = 0
        var state: LMOutput.State?
        if let lent = await lend(for: tokens) {
            (entry, read) = lent
            state = entry.state
        } else {
            let cache = try model.newCache(parameters: nil)
            guard cache.allSatisfy({ $0.isTrimmable || $0 is MambaCache }) else { return nil }
            entry = PromptCache.Entry(cache: cache)
        }
        let reused = read
        for mark in marks where mark > read {
            state = Self.prefill(tokens[read..<mark], into: entry.cache, state: state,
                                 model: model, step: prefillStep)
            entry.checkpoint(at: mark, of: tokens, state: state)
            read = mark
        }
        var rest = MLXArray(tokens[read...].map { Int32($0) })
        if input.text.tokens.ndim == 2 { rest = rest.reshaped(1, -1) }
        return Resumed(entry: entry, rest: LMInput(tokens: rest), state: state, reused: reused)
    }

    /// Run the model over `tokens`, into `cache`, a chunk at a time so a long
    /// conversation does not become one enormous activation.
    static func prefill(_ tokens: ArraySlice<Int>, into cache: [any KVCache],
                        state: LMOutput.State?, model: any LanguageModel,
                        step: Int) -> LMOutput.State? {
        var state = state
        var start = tokens.startIndex
        while start < tokens.endIndex {
            let end = min(start + step, tokens.endIndex)
            let chunk = LMInput.Text(tokens: MLXArray(tokens[start..<end].map { Int32($0) }))
            let output = model(chunk[text: .newAxis], cache: cache, state: state)
            state = output.state ?? state
            eval(cache)
            start = end
        }
        return state
    }
}
