import Foundation
import MLX
import MLXLMCommon

/// What a model has already read, kept so the next prompt reads only what is
/// new.
///
/// **Without it, every call starts from nothing.** A reply re-reads the
/// system prompt and the whole conversation, every turn. That reading
/// (prefill) is where the wait before the first word goes; the words that
/// follow cost the same either way.
///
/// **Why checkpoints.** A hybrid model such as Qwen3.5 has two kinds of
/// layer. Attention layers keep a cache that can be cut back to any length.
/// Linear-attention (gated delta, Mamba) layers keep one recurrent state that
/// only moves forward: it cannot be rewound to an arbitrary token, only put
/// back to a copy saved earlier. So a prefix is usable only up to a point
/// where that state was saved, and the points worth saving are the ones the
/// next prompt repeats exactly:
///
/// - **the end of the first message**, the system prompt, which every
///   conversation shares while nothing in it changes;
/// - **the end of the last message**, before the generation prompt, which is
///   where the next turn of the same conversation picks up. Not the end of
///   the generated reply: chat templates commonly render a past reply
///   differently from how it was generated (Qwen3.5 drops the empty
///   `<think></think>` it was generated after), so tokens past this point
///   never match.
///
/// A saved state is a reference to the arrays, not a copy. `MambaCache`, which
/// every hybrid in mlx-swift-lm uses for its recurrent layers, has its arrays
/// replaced on every step rather than written into, so what was saved stays
/// what it was. The attention layers are only cut back, never restored, so
/// they need no copy either.
///
/// **Exact or not at all.** A cache that cannot be put back exactly - see
/// `Entry.restores(_:)` - is never kept, and its prompt simply runs uncached.
///
/// **One per loaded model, and nothing is shared between models.** Drop it
/// with the model: a prefix read by other weights (a new adapter, a reload)
/// would be wrong after.
public actor PromptCache {

    /// What `choose` needs to know about a stored prefix.
    public protocol Prefix {
        var tokens: [Int] { get }
        /// Positions a cache can be put back to, each one a count of tokens.
        var checkpoints: [Int] { get }
    }

    /// One layer's cache as it was at a checkpoint.
    public enum Layer {
        /// Attention: cut back to this length.
        case rewind(offset: Int)
        /// Recurrent: put these arrays back.
        case restore(state: [MLXArray], offset: Int)
    }

    /// Everything needed to put a cache back to a checkpoint. `state` is the
    /// model's own - Qwen3.5 keeps its rope position delta there - and
    /// travels with the cache it belongs to.
    public struct Saved {
        public let layers: [Layer]
        public let state: LMOutput.State?
    }

    /// A cache and what it has read, lent out whole while a prompt uses it.
    ///
    /// `@unchecked Sendable` because a cache is a set of mutable classes. It
    /// is safe only because an entry has one user at a time: `lend` takes it
    /// out of the store and nothing else can reach it until `keep`.
    public final class Entry: Prefix, Equatable, @unchecked Sendable {
        public static func == (a: Entry, b: Entry) -> Bool { a === b }

        /// The tokens the cache represents at its deepest checkpoint.
        public private(set) var tokens: [Int]
        public let cache: [any KVCache]
        /// The model's state as of the checkpoint the cache was last put
        /// back to, to be passed with the cache when generating from it.
        public private(set) var state: LMOutput.State?
        /// False once a checkpoint was taken that could not be put back
        /// exactly. `keep` lets such an entry go.
        public private(set) var restorable = true
        private var saved: [Int: Saved]
        public var checkpoints: [Int] { saved.keys.sorted() }

        public init(cache: [any KVCache]) {
            self.tokens = []
            self.cache = cache
            self.saved = [:]
        }

        /// Remember where the cache is now, as `position` tokens of `prompt`.
        public func checkpoint(at position: Int, of prompt: [Int], state: LMOutput.State?) {
            if !cache.allSatisfy(Self.restores) { restorable = false }
            saved[position] = Saved(
                layers: cache.map { layer in
                    layer.isTrimmable
                        ? .rewind(offset: layer.offset)
                        : .restore(state: layer.state, offset: layer.offset)
                },
                state: state)
            tokens = Array(prompt.prefix(position))
        }

        /// Put the cache back to a checkpoint, and forget any deeper ones:
        /// they describe tokens this prompt does not have.
        public func restore(to position: Int) {
            guard let point = saved[position] else { return }
            for (layer, was) in zip(cache, point.layers) {
                switch was {
                case .rewind(let offset):
                    let over = layer.offset - offset
                    if over > 0 { _ = layer.trim(over) }
                case .restore(let state, let offset):
                    // Every cache is a class; `var` only because the
                    // protocol is not class-bound.
                    var layer = layer
                    layer.state = state
                    (layer as? BaseKVCache)?.offset = offset
                }
            }
            saved = saved.filter { $0.key <= position }
            tokens = Array(tokens.prefix(position))
            state = point.state
        }

        /// Whether this layer can be put back exactly, as it is now.
        ///
        /// - **Attention**: cut back, always exact.
        /// - **`MambaCache` with both slots filled**: its arrays are put back.
        ///   Not with a slot empty: `state` leaves empty slots out and setting
        ///   it sizes the cache to what it is given, so the cache would come
        ///   back one slot short and the next write to the missing one traps
        ///   (ml-explore/mlx-swift-lm#510 found this).
        /// - **Anything else that cannot be cut back** - a `CacheList` pairing
        ///   a `MambaCache` with attention, a sliding window that has wrapped -
        ///   may write into the very arrays a checkpoint would hold, so it is
        ///   refused rather than trusted.
        public static func restores(_ layer: any KVCache) -> Bool {
            if layer.isTrimmable { return true }
            if layer is MambaCache { return layer.state.count == 2 }
            return false
        }

        /// The deepest checkpoint, where the next prompt most likely resumes.
        public func rewindToDeepest() {
            if let deepest = checkpoints.last { restore(to: deepest) }
        }
    }

    /// How many prefixes are kept. See `keeping`.
    private let limit: Int
    /// The longest prefix worth keeping, in tokens. A conversation past it
    /// is read afresh each turn rather than held. On Qwen3.5-27B a token
    /// costs about 64 KB in the attention layers, so the default 12,288 is
    /// 0.8 GB a prefix.
    private let longest: Int
    /// Newest first.
    private var entries: [Entry] = []

    /// - Parameters:
    ///   - limit: prefixes kept at once. Two by default, so two prompts that
    ///     take turns on one model - a short classifier prompt and the
    ///     conversation itself - do not evict each other every turn.
    ///   - longest: the longest prefix kept, in tokens.
    public init(limit: Int = 2, longest: Int = 12_288) {
        self.limit = limit
        self.longest = longest
    }

    /// How many tokens of prefix are held, newest first.
    public var held: [Int] { entries.map(\.tokens.count) }

    /// The stored prefix this prompt can resume from, taken out of the store
    /// and put back to its checkpoint - or nil, and the prompt is read from
    /// the start into a new cache.
    public func lend(for tokens: [Int]) -> (entry: Entry, position: Int)? {
        guard let choice = Self.choose(for: tokens, among: entries) else { return nil }
        let entry = entries.remove(at: choice.entry)
        entry.restore(to: choice.position)
        return (entry, choice.position)
    }

    /// Give a prefix back once the generation that borrowed it has run to
    /// its end. **Not after a cancelled one**: it may still be stepping the
    /// model on these caches, so drop the entry instead.
    public func keep(_ entry: Entry) {
        guard entry.restorable else { return }
        entry.rewindToDeepest()
        guard entry.tokens.count <= longest else { return }
        entries = Self.keeping(entries, adding: entry, limit: limit)
    }

    /// Forget everything, for memory the system wants back.
    public func clear() {
        entries = []
    }

    // MARK: - Deciding, with no model involved

    /// Where a prompt can be checkpointed: after its first message and after
    /// its last, and never at its very end, which would leave nothing to
    /// start generating from.
    ///
    /// A message ends at an end-of-turn token, and at the newline after it
    /// when the template writes one - that newline is part of what the next
    /// prompt repeats.
    public static func checkpoints(in tokens: [Int], endOfTurn: Set<Int>,
                                   newline: Int?) -> [Int] {
        var ends: [Int] = []
        for (index, token) in tokens.enumerated() where endOfTurn.contains(token) {
            let next = index + 1
            ends.append(next < tokens.count && tokens[next] == newline ? next + 1 : next)
        }
        let usable = ends.filter { $0 < tokens.count }
        guard let first = usable.first, let last = usable.last else { return [] }
        return first == last ? [first] : [first, last]
    }

    /// The deepest checkpoint of any stored prefix that this prompt begins
    /// with, leaving at least one token to read.
    public static func choose<P: Prefix>(for tokens: [Int],
                                         among stored: [P]) -> (entry: Int, position: Int)? {
        var best: (entry: Int, position: Int)?
        for (index, prefix) in stored.enumerated() {
            for position in prefix.checkpoints.reversed()
            where position < tokens.count && position <= prefix.tokens.count {
                guard position > (best?.position ?? 0) else { break }
                if tokens[..<position].elementsEqual(prefix.tokens[..<position]) {
                    best = (index, position)
                    break
                }
            }
        }
        return best
    }

    /// The ids kept once `id` is added: newest first, at most `limit`.
    public static func keeping<ID: Equatable>(_ kept: [ID], adding id: ID,
                                              limit: Int) -> [ID] {
        Array(([id] + kept.filter { $0 != id }).prefix(limit))
    }
}
