# HybridPromptCache

A prompt cache for [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
built for hybrid models, the ones that mix attention layers with
linear-attention (gated delta, Mamba) layers, such as Qwen3.5 and Qwen3-Next.
Tested on Qwen3.5.

Each follow-up turn in a conversation reads only what is new. Measured on a
4-bit 27B `qwen3_5` checkpoint on an M5 Max with 36 GB, over a five-turn
conversation with a pasted document, the first token of a follow-up answer
arrived in a median **0.75 s instead of 11.85 s**.

## The problem

Without a cache, every call re-reads the system prompt and the whole
conversation. That reading (prefill) is where the wait before the first word
goes.

`ChatSession` in mlx-swift-lm keeps a cache across the turns of one session.
If the new prompt extends what the cache holds, it reads only the rest. If it
doesn't, it cuts the cache back to the common prefix, but only when every
layer's cache can be cut back. Otherwise it reads everything again.

On a hybrid model, that fallback happens every turn:

- The linear-attention layers keep one recurrent state that only moves
  forward. It can't be cut back to an arbitrary token.
- The chat template re-renders the previous reply differently from how it was
  generated. Qwen3.5's template generates after `<think>\n\n</think>\n\n`
  (thinking off) or `<think>\n` and the reasoning (thinking on), and drops the
  whole block when the reply comes back as history. So the next prompt never
  extends the cache: it diverges just after the last question.

Counted on a tiny Qwen3.5 through `ChatSession`, a three-turn conversation
reads 40, 64 and 88 prompt tokens: every turn, the whole prompt. The same
conversation on a pure-attention model reads 40, 30 and 30. With this package
the Qwen3.5 reads 40, 33 and 33.

## How this works

It saves the cache at **checkpoints**, the two places the next prompt repeats
exactly:

- **the end of the first message**, the system prompt, which every
  conversation shares;
- **the end of the last message**, before the generation prompt, which is
  where the next turn of the same conversation picks up.

At a checkpoint, attention layers record their length and recurrent layers
record their state. Putting a cache back cuts the attention layers back to
that length and puts the recurrent arrays back. The saved arrays are
references, not copies. That is safe because `MambaCache` has its arrays
replaced on every step rather than written into.

A new prompt borrows the stored cache with the deepest checkpoint it begins
with. It reads the rest of its messages, saving a checkpoint at each boundary,
and generates from there. When generation has run to its end, the cache is
given back. Two prefixes are kept by default, so a short classifier prompt and
the conversation can take turns on one model without evicting each other.

**It restores exactly or not at all.** Attention caches are cut back, and a
`MambaCache` with both slots filled has its arrays put back. Every hybrid in
mlx-swift-lm uses `MambaCache` for its recurrent layers. Anything else that
can't be cut back is refused, and the prompt runs uncached. That covers
FalconH1's `CacheList`, a sliding window that has wrapped, and a `MambaCache`
with an empty slot, which `state` would bring back one slot short. Only
Qwen3.5 (text and vision) is tested. Other `MambaCache` hybrids should behave
the same, but `ConversationTests`' exact-restore test is the thing to run
against a tiny config of one before relying on it.

It also carries the model's `LMOutput.State` with each checkpoint. Qwen3.5
keeps its rope position delta there and won't continue a warm cache without
it.

## Use

```swift
import HybridPromptCache
import MLXLMCommon

// One per loaded model. Drop it with the model: a prefix read by other
// weights would be wrong after.
let prompts = PromptCache()

// `Chat.Message` is not Sendable, so the conversation goes into `perform` as
// plain messages: [["role": "user", "content": "..."], ...].
let messages: [Message] = conversation

try await container.perform { context in
    let input = try await context.processor.prepare(
        input: UserInput(messages: messages))
    let parameters = GenerateParameters(maxTokens: 512, temperature: 0)

    let resumed = try await prompts.resume(
        input, model: context.model,
        boundaries: MessageBoundaries(tokenizer: context.tokenizer))

    let stream = try generateTokens(
        input: resumed?.rest ?? input, cache: resumed?.entry.cache,
        state: resumed?.state, parameters: parameters, context: context)
    for await piece in stream {
        // ...
    }
    // Only once the stream has run to its end. A cancelled generation may
    // still be writing to the cache, so let the entry go instead.
    if let resumed { await prompts.keep(resumed.entry) }
}
```

`resume` returns nil, and you generate the ordinary way, when the prompt:

- carries an image or video, since image features aren't tokens and can't be
  compared token by token;
- has a padding mask;
- is a single message, which has nothing to offer the next prompt;
- goes to a model with a cache layer that can be neither cut back nor put
  back exactly.

`MessageBoundaries(tokenizer:)` knows the end-of-turn tokens of ChatML (Qwen),
Llama 3, Gemma and Phi. For other templates, pass the ids yourself.

Call `clear()` when the system asks for memory back. On Qwen3.5-27B a cached
token costs about 64 KB in the attention layers, and each checkpoint holds the
48 recurrent states at about 3 MB apiece. Prefixes longer than 12,288 tokens
(0.8 GB) aren't kept by default.

## Tests

```sh
xcodebuild test -scheme HybridPromptCache -destination 'platform=macOS' \
    -skipPackagePluginValidation -skipMacroValidation
```

`swift test` alone can't run the model tests, because MLX needs a Metal
library that only an Xcode build produces.

The model tests use two tiny random-weight Qwen3.5s, the vision one and the
text one. Each has two attention layers and two gated-delta layers. They need
no download and finish in about a minute. What they check:

- **Restoring is exact.** Two stores see the same generated conversations.
  After each prompt, one store's cache runs further past the end than the
  other's before it is given back. The next prompt's logits match to the bit,
  over 226 prompts.
- **A cached prompt agrees with an uncached one** within upstream's own bar
  for a warm continuation: ten times the drift of reading the same prompt
  token by token, or 1e-3. The worst case measured 0.0029, where the
  token-by-token drift alone was 0.0047.
- **Cached conversations generate the same tokens as uncached ones**, through
  `generateTokens`, over four turns. Each turn resumes from the end of the
  previous question.
- **Upstream re-reads a hybrid every turn and this doesn't.** The token counts
  above come from `ChatSessionComparisonTests`. If `ChatSession` learns to
  resume hybrids, that test fails, which means this package is no longer
  needed.

The choosing logic also has property tests over thousands of generated
prompts. Each is checked against a brute-force answer, and a failure prints
the seed that reproduces it.

## Status

This depends on mlx-swift-lm `main`, because passing `state:` to
`generateTokens` came after the latest tag (3.31.4). It will move to a version
once one is tagged.

The code came out of Cider, a personal assistant that runs its models on the
Mac and iPhone. Related upstream work:
[#84](https://github.com/ml-explore/mlx-swift-lm/issues/84) (prompt cache
feature request),
[#436](https://github.com/ml-explore/mlx-swift-lm/pull/436) (cross-request
LRU prompt cache), and
[#510](https://github.com/ml-explore/mlx-swift-lm/pull/510) (rewindable
`MambaCache`).

## License

MIT
