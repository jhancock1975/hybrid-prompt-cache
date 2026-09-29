import MLX
import MLXLMCommon
import XCTest
@testable import HybridPromptCache

/// Exact or not at all: which caches may be put back, and what happens to an
/// entry holding one that may not.
final class RestorableTests: XCTestCase {

    private func mamba(_ slots: [Bool]) -> MambaCache {
        let cache = MambaCache()
        for (index, filled) in slots.enumerated() where filled {
            cache[index] = MLXArray.zeros([1, 4])
        }
        return cache
    }

    func testAttentionAndAFullMambaCacheAreRestorable() {
        let entry = PromptCache.Entry(cache: [KVCacheSimple(), mamba([true, true])])
        entry.checkpoint(at: 3, of: [1, 2, 3, 4], state: nil)
        XCTAssertTrue(entry.restorable)
    }

    /// Put back through `state`, a `MambaCache` with its first slot empty
    /// would come back with one slot, and the next write to the second one
    /// would trap. So it is not kept.
    func testAMambaCacheWithAnEmptySlotIsNotKept() async {
        let entry = PromptCache.Entry(cache: [KVCacheSimple(), mamba([false, true])])
        entry.checkpoint(at: 3, of: [1, 2, 3, 4], state: nil)
        XCTAssertFalse(entry.restorable)

        let prompts = PromptCache()
        await prompts.keep(entry)
        let held = await prompts.held
        XCTAssertEqual(held, [])
    }

    /// FalconH1 pairs a `MambaCache` with attention in one `CacheList`, which
    /// can be neither cut back nor trusted to be put back.
    func testACacheListIsNotKept() async {
        let layer = CacheList(mamba([true, true]), KVCacheSimple())
        XCTAssertFalse(layer.isTrimmable)
        let entry = PromptCache.Entry(cache: [layer])
        entry.checkpoint(at: 3, of: [1, 2, 3, 4], state: nil)
        XCTAssertFalse(entry.restorable)

        let prompts = PromptCache()
        await prompts.keep(entry)
        let held = await prompts.held
        XCTAssertEqual(held, [])
    }
}
