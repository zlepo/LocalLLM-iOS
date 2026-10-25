import Foundation
import Combine
import llama

@MainActor
public final class LLMEngine: ObservableObject {
public static let shared = LLMEngine()

@Published public private(set) var isModelLoaded = false
@Published public private(set) var isGenerating = false
@Published public private(set) var tokensPerSecond: Double = 0
@Published public private(set) var modelStatus = "No model imported"
@Published public private(set) var loadedModelName: String?
@Published public private(set) var lastError: String?

private static let savedModelPathKey = "LocalLLM.savedModelPath"

private init() {
let testTime = llama_time_us()
print("llama.cpp API test: \(testTime)")
}

private var model: OpaquePointer?
private var context: OpaquePointer?
private var vocab: OpaquePointer?

public var savedModelPath: String? {
guard let path = UserDefaults.standard.string(forKey: Self.savedModelPathKey),
      FileManager.default.fileExists(atPath: path) else {
    return nil
}
return path
}

public func importModel(from sourceURL: URL) async throws {
guard sourceURL.pathExtension.lowercased() == "gguf" else {
    let error = NSError(
        domain: "LocalLLM",
        code: 10,
        userInfo: [NSLocalizedDescriptionKey: "Selected file is not a GGUF model"]
    )
    lastError = error.localizedDescription
    modelStatus = "Invalid model file"
    throw error
}

let hasAccess = sourceURL.startAccessingSecurityScopedResource()
defer {
    if hasAccess {
        sourceURL.stopAccessingSecurityScopedResource()
    }
}

modelStatus = "Importing \(sourceURL.lastPathComponent)..."
lastError = nil

do {
    let destination = try await Task.detached(priority: .userInitiated) {
        let fileManager = FileManager.default
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        let modelsDirectory = appSupport.appendingPathComponent("Models", isDirectory: true)
        try fileManager.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )

        let destination = modelsDirectory.appendingPathComponent(sourceURL.lastPathComponent)
        let temporary = modelsDirectory.appendingPathComponent(UUID().uuidString + ".gguf.tmp")

        if fileManager.fileExists(atPath: temporary.path) {
            try fileManager.removeItem(at: temporary)
        }

        do {
            try fileManager.copyItem(at: sourceURL, to: temporary)

            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }

            try fileManager.moveItem(at: temporary, to: destination)
            return destination
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw error
        }
    }.value

    modelStatus = "Import complete. Loading model..."
    try await loadModel(from: destination.path)
} catch {
    lastError = error.localizedDescription
    modelStatus = "Import failed: \(error.localizedDescription)"
    throw error
}
}
public func loadSavedModelIfAvailable() async {
guard let path = savedModelPath else {
    modelStatus = "No model imported"
    return
}

do {
    try await loadModel(from: path)
} catch {
    lastError = error.localizedDescription
    modelStatus = "Failed to load saved model"
}
}

public func loadModel(from path: String) async throws {
guard FileManager.default.fileExists(atPath: path) else {
    let error = NSError(
        domain: "LocalLLM",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "GGUF file does not exist at the saved path"]
    )
    lastError = error.localizedDescription
    modelStatus = "Model file missing"
    throw error
}

modelStatus = "Loading model..."
lastError = nil
isModelLoaded = false
loadedModelName = nil
if let context {
llama_free(context)
self.context = nil
}

if let model {
llama_model_free(model)
self.model = nil
self.vocab = nil
}

var modelParams = llama_model_default_params()

guard let loadedModel = llama_model_load_from_file(path, modelParams) else {
throw NSError(
domain: "LocalLLM",
code: 2,
userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF model"]
)
}

var contextParams = llama_context_default_params()
contextParams.n_ctx = 2048
contextParams.n_batch = 512

guard let loadedContext = llama_init_from_model(
loadedModel,
contextParams
) else {
llama_model_free(loadedModel)

throw NSError(
domain: "LocalLLM",
code: 3,
userInfo: [NSLocalizedDescriptionKey: "Failed to create llama context"]
)
}

self.model = loadedModel
self.context = loadedContext
self.vocab = llama_model_get_vocab(loadedModel)
self.isModelLoaded = true
self.loadedModelName = URL(fileURLWithPath: path).lastPathComponent
self.modelStatus = "Loaded: \(self.loadedModelName ?? "GGUF model")"
self.lastError = nil
UserDefaults.standard.set(path, forKey: Self.savedModelPathKey)

print("GGUF loaded successfully with real llama.cpp context")
}

public func generate(
prompt: String,
systemPrompt: String? = nil,
maxTokens: Int = 1024,
stopSequences: [String] = []
) async throws -> String {
guard let context = self.context,
      let model = self.model,
      let vocab = self.vocab else {
    throw NSError(
        domain: "LocalLLM",
        code: 4,
        userInfo: [NSLocalizedDescriptionKey: "No GGUF model is loaded"]
    )
}

isGenerating = true
defer { isGenerating = false }

let fullPrompt: String

if let template = llama_model_chat_template(model, nil) {
    let systemRole = strdup("system")
    let userRole = strdup("user")
    let systemContent = systemPrompt.map { strdup($0) }
    let userContent = strdup(prompt)

    defer {
        free(systemRole)
        free(userRole)
        if let systemContent { free(systemContent) }
        free(userContent)
    }

    var messages: [llama_chat_message] = []

    if let systemContent {
        messages.append(
            llama_chat_message(
                role: UnsafePointer(systemRole),
                content: UnsafePointer(systemContent)
            )
        )
    }

    messages.append(
        llama_chat_message(
            role: UnsafePointer(userRole),
            content: UnsafePointer(userContent)
        )
    )

    var chatBuffer = [CChar](repeating: 0, count: max(4096, prompt.utf8.count * 2 + 1024))

    var formattedLength = messages.withUnsafeBufferPointer { messageBuffer in
        llama_chat_apply_template(
            template,
            messageBuffer.baseAddress,
            messages.count,
            true,
            &chatBuffer,
            Int32(chatBuffer.count)
        )
    }

    if formattedLength < 0 {
        throw NSError(
            domain: "LocalLLM",
            code: 5,
            userInfo: [NSLocalizedDescriptionKey: "Failed to apply model chat template"]
        )
    }

    if formattedLength >= chatBuffer.count {
        chatBuffer = [CChar](repeating: 0, count: Int(formattedLength) + 1)

        formattedLength = messages.withUnsafeBufferPointer { messageBuffer in
            llama_chat_apply_template(
                template,
                messageBuffer.baseAddress,
                messages.count,
                true,
                &chatBuffer,
                Int32(chatBuffer.count)
            )
        }

        guard formattedLength >= 0,
              formattedLength < chatBuffer.count else {
            throw NSError(
                domain: "LocalLLM",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Failed to resize chat-template buffer"]
            )
        }
    }

    fullPrompt = String(
        decoding: chatBuffer.prefix(Int(formattedLength)).map { UInt8(bitPattern: $0) },
        as: UTF8.self
    )
} else if let systemPrompt, !systemPrompt.isEmpty {
    fullPrompt = "\(systemPrompt)\n\n\(prompt)"
} else {
    fullPrompt = prompt
}

let utf8Count = fullPrompt.utf8.count
var tokenCapacity = max(utf8Count + 32, 128)
var tokens = [llama_token](repeating: 0, count: tokenCapacity)

var tokenCount = llama_tokenize(
    vocab,
    fullPrompt,
    Int32(utf8Count),
    &tokens,
    Int32(tokens.count),
    true,
    true
)

if tokenCount < 0 {
    tokenCapacity = Int(-tokenCount)
    tokens = [llama_token](repeating: 0, count: tokenCapacity)

    tokenCount = llama_tokenize(
        vocab,
        fullPrompt,
        Int32(utf8Count),
        &tokens,
        Int32(tokens.count),
        true,
        true
    )
}

guard tokenCount > 0 else {
    throw NSError(
        domain: "LocalLLM",
        code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Failed to tokenize prompt"]
    )
}

tokens.removeSubrange(Int(tokenCount)..<tokens.count)

let contextSize = Int(llama_n_ctx(context))
let requestedOutput = max(1, maxTokens)
let availableOutput = contextSize - tokens.count

guard availableOutput > 0 else {
    throw NSError(
        domain: "LocalLLM",
        code: 7,
        userInfo: [NSLocalizedDescriptionKey: "Prompt is too large for the current context window"]
    )
}

let generationLimit = min(requestedOutput, availableOutput)

llama_memory_clear(llama_get_memory(context), true)

let batchCapacity = min(512, max(tokens.count, 1))
var batch = llama_batch_init(Int32(batchCapacity), 0, 1)
defer { llama_batch_free(batch) }

var promptOffset = 0

while promptOffset < tokens.count {
    let chunkCount = min(batchCapacity, tokens.count - promptOffset)
    batch.n_tokens = Int32(chunkCount)

    for i in 0..<chunkCount {
        batch.token[i] = tokens[promptOffset + i]
        batch.pos[i] = Int32(promptOffset + i)
        batch.n_seq_id[i] = 1
        batch.seq_id[i]![0] = 0
        batch.logits[i] = (promptOffset + i == tokens.count - 1) ? 1 : 0
    }

    guard llama_decode(context, batch) == 0 else {
        throw NSError(
            domain: "LocalLLM",
            code: 8,
            userInfo: [NSLocalizedDescriptionKey: "Failed while decoding prompt"]
        )
    }

    promptOffset += chunkCount
}

let samplerParams = llama_sampler_chain_default_params()

guard let sampler = llama_sampler_chain_init(samplerParams) else {
    throw NSError(
        domain: "LocalLLM",
        code: 9,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create sampler"]
    )
}

defer { llama_sampler_free(sampler) }

llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7))
llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max)))

var output = ""
var position = Int32(tokens.count)

for _ in 0..<generationLimit {
    let token = llama_sampler_sample(
        sampler,
        context,
        batch.n_tokens - 1
    )

    if llama_vocab_is_eog(vocab, token) {
        break
    }

    var pieceBuffer = [CChar](repeating: 0, count: 256)

    var pieceLength = llama_token_to_piece(
        vocab,
        token,
        &pieceBuffer,
        Int32(pieceBuffer.count),
        0,
        true
    )

    if pieceLength < 0 {
        pieceBuffer = [CChar](repeating: 0, count: Int(-pieceLength))

        pieceLength = llama_token_to_piece(
            vocab,
            token,
            &pieceBuffer,
            Int32(pieceBuffer.count),
            0,
            true
        )
    }

    if pieceLength > 0 {
        let bytes = pieceBuffer.prefix(Int(pieceLength)).map {
            UInt8(bitPattern: $0)
        }

        output += String(decoding: bytes, as: UTF8.self)
    }

    if stopSequences.contains(where: { !$0.isEmpty && output.hasSuffix($0) }) {
        break
    }

    batch.n_tokens = 1
    batch.token[0] = token
    batch.pos[0] = position
    batch.n_seq_id[0] = 1
    batch.seq_id[0]![0] = 0
    batch.logits[0] = 1

    guard llama_decode(context, batch) == 0 else {
        throw NSError(
            domain: "LocalLLM",
            code: 10,
            userInfo: [NSLocalizedDescriptionKey: "Failed while generating token"]
        )
    }

    position += 1
}

return output.trimmingCharacters(in: .whitespacesAndNewlines)
}
}
