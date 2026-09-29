import MLX
import MLXLMCommon
import XCTest
import HybridPromptCache

/// The README's example, as written there, through a real `ModelContainer`:
/// if it stops compiling or stops resuming, so does this.
final class ReadmeExampleTests: XCTestCase {

    func testTheReadmeExampleResumesTheSecondTurn() async throws {
        let container = ModelContainer(context: Tiny.context(try Tiny.qwen35()))
        let prompts = PromptCache()
        var reused: [Int] = []

        let system: Message = ["role": "system", "content": "you answer questions"]
        let question: Message = ["role": "user", "content": "what colour is the door"]
        for messages: [Message] in [
            [system, question],
            [system, question, ["role": "assistant", "content": "w20 w21"],
             ["role": "user", "content": "and the windows"]],
        ] {
            // From here to the closing brace is the README's example.
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
                    _ = piece
                }
                // Only once the stream has run to its end. A cancelled generation may
                // still be writing to the cache, so let the entry go instead.
                if let resumed { await prompts.keep(resumed.entry) }
            }
            // Not the README's: how the test sees what happened.
            reused.append(await prompts.held.first ?? 0)
        }
        XCTAssertEqual(reused.count, 2)
        XCTAssertGreaterThan(reused[1], reused[0], "the second turn kept nothing deeper")
    }
}
