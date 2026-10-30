import Foundation
import Combine

/// AgentOrchestrator manages sub-agents and routes tasks
/// This is where the magic happens - parallel task execution, tool routing, and prompt chaining
@MainActor
public final class AgentOrchestrator: ObservableObject {
    public static let shared = AgentOrchestrator()
    
    // MARK: - Published State
    
    @Published public private(set) var activeAgentCount: Int = 0
    @Published public private(set) var runningTasks: [AgentTask] = []
    @Published public private(set) var taskHistory: [CompletedTask] = []
    /// Short progress label shown while a reply is being produced (e.g. "Searching the web").
    @Published public private(set) var statusText: String?
    
    // MARK: - Dependencies
    
    private let llm = LLMEngine.shared
    private let mcp = MCPManager.shared
    
    // MARK: - Configuration
    
    public struct Config {
        var maxConcurrentAgents: Int = 4
        var maxChainDepth: Int = 10
        var defaultTimeout: TimeInterval = 60
        var enableParallelExecution: Bool = true
    }
    
    private var config = Config()
    
    // MARK: - Types
    
    public struct AgentTask: Identifiable {
        public let id: UUID
        public let name: String
        public let prompt: String
        public let parentID: UUID?
        public var status: Status
        public let startTime: Date
        
        public enum Status {
            case pending
            case running
            case completed(String)
            case failed(Error)
        }
    }
    
    public struct CompletedTask: Identifiable {
        public let id: UUID
        public let name: String
        public let result: String
        public let duration: TimeInterval
        public let toolsUsed: [String]
    }
    
    // MARK: - Initialization
    
    private init() {}
    
    // MARK: - Main Entry Point
    
    /// Process a user prompt, deciding whether to use tools or spawn sub-agents
    func process(
        prompt: String,
        conversation: [Message] = []
    ) async throws -> String {
        // Normal chat goes directly to the model, with web results added for factual questions.
        // Agent/tool routing remains available below for future MCP use.
        defer { statusText = nil }

        var sources: [WebSearch.Result] = []
        var searchedQuery: String?
        if WebSearch.isEnabled {
            statusText = "Checking if this needs a search"
            if let query = await searchQuery(for: prompt, conversation: conversation) {
                statusText = "Searching the web"
                searchedQuery = query
                sources = await WebSearch.search(query)
                appendLlamaLog("[LocalLLM] web search \"\(query)\" -> \(sources.count) results\n")
            }
        }

        let sourcesBlock = formatSources(sources, query: searchedQuery)
        let context = buildContext(conversation, reserveChars: sourcesBlock.count)

        var sections: [String] = []
        if !context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sections.append("CONVERSATION:\n\(context)")
        }
        if !sourcesBlock.isEmpty {
            sections.append(sourcesBlock)
        }

        let directPrompt: String
        if sections.isEmpty {
            directPrompt = prompt
        } else {
            directPrompt = sections.joined(separator: "\n\n") + "\n\nCURRENT USER MESSAGE:\n\(prompt)"
        }

        statusText = sources.isEmpty ? "Thinking" : "Reading \(sources.count) sources"
        let answer = try await llm.generate(
            prompt: directPrompt,
            systemPrompt: systemPrompt,
            maxTokens: 1024
        )

        return answer + sourceFooter(for: answer, sources: sources)
    }

    // MARK: - Web Search Routing

    /// Ask the model whether the message needs a web lookup, and for a search query if so.
    private func searchQuery(for prompt: String, conversation: [Message]) async -> String? {
        let recent = buildContext(conversation, maxChars: 1500)
        let routerPrompt = """
        \(recent.isEmpty ? "" : "RECENT CONVERSATION:\n\(recent)\n\n")LATEST USER MESSAGE:
        \(prompt)
        """
        let routerSystem = """
        You decide whether answering the latest user message needs a web search.
        Search for anything factual you could misremember: dates, years, names, numbers, specs, \
        prices, people, products, places, events, news, or anything recent.
        Do not search for greetings, small talk, opinions, creative writing, math, or code.
        Reply with exactly one line and nothing else:
        SEARCH: <short web search query>
        or
        NO_SEARCH
        """

        let reply: String
        do {
            reply = try await llm.generate(prompt: routerPrompt, systemPrompt: routerSystem, maxTokens: 40)
        } catch {
            return nil
        }

        for line in reply.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.uppercased().hasPrefix("NO_SEARCH") { return nil }
            if trimmed.uppercased().hasPrefix("SEARCH:") {
                let query = trimmed.dropFirst("SEARCH:".count)
                    .trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
                return query.isEmpty ? nil : String(query.prefix(200))
            }
        }

        // Unclear router output: lean towards searching real questions rather than guessing.
        return prompt.contains("?") ? String(prompt.prefix(200)) : nil
    }

    private func formatSources(_ sources: [WebSearch.Result], query: String?) -> String {
        guard !sources.isEmpty else { return "" }
        let entries = sources.enumerated().map { index, source in
            "[\(index + 1)] \(source.title) (\(source.url))\n\(source.text)"
        }
        return """
        WEB RESULTS for "\(query ?? "")":
        \(entries.joined(separator: "\n\n"))

        Answer using these web results where they are relevant and cite them like [1]. \
        If the results do not contain the answer, say you could not find it instead of guessing.
        """
    }

    /// List the sources the answer cited (or the top few if it cited none) so the user can check them.
    private func sourceFooter(for answer: String, sources: [WebSearch.Result]) -> String {
        guard !sources.isEmpty else { return "" }
        var cited = sources.indices.filter { answer.contains("[\($0 + 1)]") }
        if cited.isEmpty { cited = Array(sources.indices.prefix(3)) }
        let lines = cited.map { "[\($0 + 1)] \(sources[$0].url)" }
        return "\n\nSources:\n" + lines.joined(separator: "\n")
    }
    // MARK: - Execution Strategies
    
    private func executeBasedOnAnalysis(
        analysis: String,
        originalPrompt: String,
        context: String
    ) async throws -> String {
        
        // Parse approach from analysis
        let approach = parseApproach(from: analysis)
        
        switch approach {
        case .direct:
            return try await directResponse(prompt: originalPrompt, context: context)
            
        case .tool(let toolCalls):
            return try await executeToolSequence(toolCalls, originalPrompt: originalPrompt)
            
        case .parallel(let subtasks):
            return try await executeParallel(subtasks, originalPrompt: originalPrompt)
            
        case .unknown:
            // Fall back to direct response
            return try await directResponse(prompt: originalPrompt, context: context)
        }
    }
    
    /// Direct LLM response without tools
    private func directResponse(prompt: String, context: String) async throws -> String {
        return try await llm.generate(
            prompt: prompt,
            systemPrompt: systemPrompt + "\n\nCONTEXT:\n\(context)",
            maxTokens: 2048
        )
    }
    
    /// Execute tools in sequence, chaining results
    private func executeToolSequence(
        _ toolCalls: [MCPManager.ToolCall],
        originalPrompt: String
    ) async throws -> String {
        
        var results: [(tool: String, result: String)] = []
        
        for call in toolCalls {
            let task = AgentTask(
                id: UUID(),
                name: "Tool: \(call.toolName)",
                prompt: "",
                parentID: nil,
                status: .running,
                startTime: Date()
            )
            runningTasks.append(task)
            activeAgentCount += 1
            
            do {
                let result = try await mcp.callTool(call)
                let resultString = String(describing: result.content)
                results.append((call.toolName, resultString))
                
                // Update task status
                if let index = runningTasks.firstIndex(where: { $0.id == task.id }) {
                    runningTasks[index].status = .completed(resultString)
                }
            } catch {
                if let index = runningTasks.firstIndex(where: { $0.id == task.id }) {
                    runningTasks[index].status = .failed(error)
                }
                throw error
            }
            
            activeAgentCount -= 1
        }
        
        // Synthesize results
        let synthesisPrompt = """
        The user asked: \(originalPrompt)
        
        Tool results:
        \(results.map { "[\($0.tool)]: \($0.result)" }.joined(separator: "\n\n"))
        
        Synthesize these results into a helpful response for the user.
        """
        
        return try await llm.generate(
            prompt: synthesisPrompt,
            systemPrompt: systemPrompt,
            maxTokens: 2048
        )
    }
    
    /// Execute subtasks in parallel using sub-agents
    private func executeParallel(
        _ subtasks: [Subtask],
        originalPrompt: String
    ) async throws -> String {
        
        // Limit concurrency
        let limitedSubtasks = Array(subtasks.prefix(config.maxConcurrentAgents))
        
        // Create tasks for tracking
        let tasks = limitedSubtasks.map { subtask in
            AgentTask(
                id: UUID(),
                name: subtask.name,
                prompt: subtask.prompt,
                parentID: nil,
                status: .pending,
                startTime: Date()
            )
        }
        runningTasks.append(contentsOf: tasks)
        activeAgentCount += tasks.count
        
        // Execute in parallel
        let results = await withTaskGroup(of: (Int, Result<String, Error>).self) { group in
            for (index, subtask) in limitedSubtasks.enumerated() {
                group.addTask {
                    do {
                        let result = try await self.executeSubAgent(
                            name: subtask.name,
                            prompt: subtask.prompt
                        )
                        return (index, .success(result))
                    } catch {
                        return (index, .failure(error))
                    }
                }
            }
            
            var results: [(Int, Result<String, Error>)] = []
            for await result in group {
                results.append(result)
            }
            return results.sorted { $0.0 < $1.0 }
        }
        
        // Update task statuses and collect results
        var successResults: [(name: String, result: String)] = []
        
        for (index, result) in results {
            let task = tasks[index]
            
            switch result {
            case .success(let response):
                if let taskIndex = runningTasks.firstIndex(where: { $0.id == task.id }) {
                    runningTasks[taskIndex].status = .completed(response)
                }
                successResults.append((limitedSubtasks[index].name, response))
                
            case .failure(let error):
                if let taskIndex = runningTasks.firstIndex(where: { $0.id == task.id }) {
                    runningTasks[taskIndex].status = .failed(error)
                }
            }
        }
        
        activeAgentCount -= tasks.count
        
        // Synthesize parallel results
        let synthesisPrompt = """
        The user asked: \(originalPrompt)
        
        I delegated to \(successResults.count) sub-agents. Their results:
        
        \(successResults.map { "## \($0.name)\n\($0.result)" }.joined(separator: "\n\n---\n\n"))
        
        Synthesize these results into a cohesive, helpful response.
        """
        
        return try await llm.generate(
            prompt: synthesisPrompt,
            systemPrompt: systemPrompt,
            maxTokens: 2048
        )
    }
    
    /// Execute a single sub-agent task
    private func executeSubAgent(name: String, prompt: String) async throws -> String {
        let subAgentPrompt = """
        You are a focused sub-agent working on a specific task.
        
        YOUR TASK: \(name)
        
        INSTRUCTIONS: \(prompt)
        
        \(mcp.toolsPrompt())
        
        Complete the task thoroughly but concisely.
        """
        
        // Sub-agents can also use tools
        var response = try await llm.generate(
            prompt: subAgentPrompt,
            systemPrompt: "You are a helpful sub-agent. Complete your assigned task.",
            maxTokens: 1024
        )
        
        // Check if sub-agent wants to use a tool
        if let toolCall = parseToolCall(from: response) {
            let toolResult = try await mcp.callTool(toolCall)
            
            // Let sub-agent incorporate tool result
            let followUp = """
            Tool result for \(toolCall.toolName):
            \(toolResult.content)
            
            Now provide your final response incorporating this information.
            """
            
            response = try await llm.generate(
                prompt: followUp,
                systemPrompt: "Incorporate the tool result and complete your task.",
                maxTokens: 1024
            )
        }
        
        return response
    }
    
    // MARK: - Parsing Helpers
    
    private enum Approach {
        case direct
        case tool([MCPManager.ToolCall])
        case parallel([Subtask])
        case unknown
    }
    
    private struct Subtask {
        let name: String
        let prompt: String
    }
    
    private func parseApproach(from analysis: String) -> Approach {
        // Extract approach from <analysis> block
        if analysis.contains("approach: DIRECT") || analysis.contains("approach:DIRECT") {
            return .direct
        }
        
        if analysis.contains("approach: PARALLEL") || analysis.contains("<subtasks>") {
            let subtasks = parseSubtasks(from: analysis)
            if !subtasks.isEmpty {
                return .parallel(subtasks)
            }
        }
        
        if analysis.contains("approach: TOOL") || analysis.contains("<tools>") {
            let toolCalls = parseToolCalls(from: analysis)
            if !toolCalls.isEmpty {
                return .tool(toolCalls)
            }
        }
        
        return .unknown
    }
    
    private func parseSubtasks(from text: String) -> [Subtask] {
        // Simple YAML-like parsing for subtasks
        var subtasks: [Subtask] = []
        
        guard let start = text.range(of: "<subtasks>"),
              let end = text.range(of: "</subtasks>") else {
            return []
        }
        
        let content = String(text[start.upperBound..<end.lowerBound])
        let lines = content.components(separatedBy: .newlines)
        
        var currentName: String?
        var currentPrompt: String?
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            if trimmed.hasPrefix("- name:") {
                // Save previous if exists
                if let name = currentName, let prompt = currentPrompt {
                    subtasks.append(Subtask(name: name, prompt: prompt))
                }
                currentName = String(trimmed.dropFirst(7)).trimmingCharacters(in: .whitespaces)
                currentPrompt = nil
            } else if trimmed.hasPrefix("prompt:") {
                currentPrompt = String(trimmed.dropFirst(7)).trimmingCharacters(in: .whitespaces)
            }
        }
        
        // Don't forget last one
        if let name = currentName, let prompt = currentPrompt {
            subtasks.append(Subtask(name: name, prompt: prompt))
        }
        
        return subtasks
    }
    
    private func parseToolCalls(from text: String) -> [MCPManager.ToolCall] {
        // Simple parsing for tool calls
        var calls: [MCPManager.ToolCall] = []
        
        guard let start = text.range(of: "<tools>"),
              let end = text.range(of: "</tools>") else {
            return []
        }
        
        let content = String(text[start.upperBound..<end.lowerBound])
        let lines = content.components(separatedBy: .newlines)
        
        var currentTool: String?
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            if trimmed.hasPrefix("- tool_name:") || trimmed.hasPrefix("tool_name:") {
                let name = trimmed.replacingOccurrences(of: "- tool_name:", with: "")
                    .replacingOccurrences(of: "tool_name:", with: "")
                    .trimmingCharacters(in: .whitespaces)
                currentTool = name
            }
            
            // For simplicity, create call when we have a tool name
            // In production, you'd parse the arguments too
            if let tool = currentTool {
                calls.append(MCPManager.ToolCall(toolName: tool, arguments: [:]))
                currentTool = nil
            }
        }
        
        return calls
    }
    
    private func parseToolCall(from response: String) -> MCPManager.ToolCall? {
        guard let start = response.range(of: "<tool_use>"),
              let end = response.range(of: "</tool_use>") else {
            return nil
        }
        
        let content = String(response[start.upperBound..<end.lowerBound])
        let lines = content.components(separatedBy: .newlines)
        
        var toolName: String?
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("name:") {
                toolName = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            }
        }
        
        guard let name = toolName else { return nil }
        return MCPManager.ToolCall(toolName: name, arguments: [:])
    }
    
    // MARK: - Context Building
    
    private func buildContext(_ conversation: [Message], reserveChars: Int = 0, maxChars: Int? = nil) -> String {
        // Keep as much recent history as fits the context window, leaving room for the
        // system prompt, web results and up to 2048 output tokens. ~3 chars/token is a safe estimate.
        let budgetTokens = max(512, LLMEngine.configuredContextSize - 2048 - 1024)
        var remainingChars = min(maxChars ?? .max, max(0, budgetTokens * 3 - reserveChars))
        var lines: [String] = []
        for msg in conversation.reversed() {
            let role = msg.role == .user ? "User" : "Assistant"
            let line = "\(role): \(msg.content)"
            if line.count > remainingChars { break }
            remainingChars -= line.count + 1
            lines.append(line)
        }
        return lines.reversed().joined(separator: "\n")
    }
    
    // MARK: - System Prompt
    
    private var systemPrompt: String {
        """
        You are a helpful AI assistant running locally on the user's device.

        Answer the user's actual question directly and clearly.
        Prefer a short, useful answer unless the user asks for more detail.
        Use prior conversation only when it is relevant to the current question.

        REASONING RULES:
        - Think through the problem, but do not expose internal chain-of-thought.
        - Do not repeatedly reconsider the same possibilities.
        - Do not invent alternate interpretations when the user's intent is clear.
        - If uncertain, state the uncertainty briefly instead of rambling.
        - Never guess facts. If you are not sure of a fact and the web results do not give it, \
        say you don't know rather than making something up.
        - Stop once the question has been answered.

        CAPABILITIES:
        - Direct knowledge and reasoning
        - Tool usage via MCP servers when connected
        - Sub-agents for genuinely independent complex tasks
        - Multi-step task execution when necessary

        Be accurate, conversational, concise, and useful.
        """
    }
}

// MARK: - Web Search

/// Web lookup for factual questions: Brave Search API when a key is set, DuckDuckGo HTML otherwise
/// (or when Brave fails). Safe search is off on both. The top pages are fetched and trimmed to the
/// passages that best match the query so the model gets more than a one-line snippet.
enum WebSearch {
    struct Result: Sendable {
        let title: String
        let url: String
        var text: String
    }

    static let enabledKey = "LocalLLM.webSearchEnabled"
    static let braveKeyKey = "LocalLLM.braveAPIKey"

    static var isEnabled: Bool {
        UserDefaults.standard.object(forKey: enabledKey) as? Bool ?? true
    }

    private static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"

    static func search(_ query: String, limit: Int = 5) async -> [Result] {
        var results: [Result] = []

        let braveKey = (UserDefaults.standard.string(forKey: braveKeyKey) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !braveKey.isEmpty {
            do {
                results = try await brave(query, key: braveKey, limit: limit)
            } catch {
                appendLlamaLog("[LocalLLM] Brave search failed: \(error.localizedDescription)\n")
            }
        }

        if results.isEmpty {
            do {
                results = try await duckDuckGo(query, limit: limit)
            } catch {
                appendLlamaLog("[LocalLLM] DuckDuckGo search failed: \(error.localizedDescription)\n")
            }
        }

        // Add the most relevant passages from the top 3 pages.
        let pages = await withTaskGroup(of: (Int, String?).self) { group -> [Int: String] in
            for (index, result) in results.prefix(3).enumerated() {
                let url = result.url
                group.addTask { (index, try? await pageExcerpt(url, query: query)) }
            }
            var collected: [Int: String] = [:]
            for await (index, text) in group {
                if let text, !text.isEmpty { collected[index] = text }
            }
            return collected
        }
        for (index, text) in pages {
            results[index].text += "\n" + text
        }

        return results
    }

    // MARK: Providers

    private static func brave(_ query: String, key: String, limit: Int) async throws -> [Result] {
        var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(limit)),
            URLQueryItem(name: "safesearch", value: "off"),
            URLQueryItem(name: "extra_snippets", value: "true")
        ]
        var request = URLRequest(url: components.url!, timeoutInterval: 10)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(key, forHTTPHeaderField: "X-Subscription-Token")

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "Brave HTTP \(http.statusCode)"])
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let web = json["web"] as? [String: Any],
              let items = web["results"] as? [[String: Any]] else {
            return []
        }

        return items.prefix(limit).compactMap { item -> Result? in
            guard let url = item["url"] as? String else { return nil }
            let title = stripTags(item["title"] as? String ?? url)
            var snippets = [stripTags(item["description"] as? String ?? "")]
            snippets += (item["extra_snippets"] as? [String] ?? []).map(stripTags)
            return Result(title: title, url: url, text: snippets.filter { !$0.isEmpty }.joined(separator: " "))
        }
    }

    private static func duckDuckGo(_ query: String, limit: Int) async throws -> [Result] {
        var request = URLRequest(url: URL(string: "https://html.duckduckgo.com/html/")!, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var form = URLComponents()
        form.queryItems = [URLQueryItem(name: "q", value: query), URLQueryItem(name: "kp", value: "-2")]
        request.httpBody = form.percentEncodedQuery?.data(using: .utf8)

        let (data, _) = try await URLSession.shared.data(for: request)
        let html = String(decoding: data, as: UTF8.self)

        let links = matches(#"class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#, in: html)
        let snippets = matches(#"class="result__snippet"[^>]*>(.*?)</a>"#, in: html)

        var results: [Result] = []
        for (index, link) in links.enumerated() {
            guard let url = resolveDuckDuckGoURL(link[0]) else { continue }
            let snippet = index < snippets.count ? stripTags(snippets[index][0]) : ""
            results.append(Result(title: stripTags(link[1]), url: url, text: snippet))
            if results.count == limit { break }
        }
        return results
    }

    /// DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded url>; ads go via y.js.
    private static func resolveDuckDuckGoURL(_ raw: String) -> String? {
        let href = decodeEntities(raw)
        if href.contains("duckduckgo.com/y.js") { return nil }
        if let components = URLComponents(string: href.hasPrefix("//") ? "https:" + href : href),
           let target = components.queryItems?.first(where: { $0.name == "uddg" })?.value {
            return target
        }
        return href.hasPrefix("http") ? href : nil
    }

    // MARK: Page text

    private static func pageExcerpt(_ urlString: String, query: String, maxChars: Int = 1200) async throws -> String? {
        guard let url = URL(string: urlString), url.scheme == "https" else { return nil }
        var request = URLRequest(url: url, timeoutInterval: 6)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              (http.value(forHTTPHeaderField: "Content-Type") ?? "").contains("html") else {
            return nil
        }

        var html = String(decoding: data.prefix(1_500_000), as: UTF8.self)
        for tag in ["script", "style", "noscript", "svg", "nav", "footer", "header"] {
            html = html.replacingOccurrences(
                of: "(?s)<\(tag)\\b[^>]*>.*?</\(tag)>",
                with: " ",
                options: [.regularExpression, .caseInsensitive]
            )
        }
        let text = stripTags(html)

        // Score ~400-char windows by how many query words they contain and keep the best ones.
        let words = Set(query.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init).filter { $0.count > 2 })
        guard !words.isEmpty else { return String(text.prefix(maxChars)) }

        let chunkSize = 400
        var chunks: [(score: Int, offset: Int, text: String)] = []
        var start = text.startIndex
        var offset = 0
        while start < text.endIndex {
            let end = text.index(start, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            let chunk = String(text[start..<end])
            let lower = chunk.lowercased()
            let score = words.reduce(0) { $0 + (lower.contains($1) ? 1 : 0) }
            if score > 0 { chunks.append((score, offset, chunk)) }
            start = end
            offset += 1
        }

        let best = chunks
            .sorted { $0.score != $1.score ? $0.score > $1.score : $0.offset < $1.offset }
            .prefix(maxChars / chunkSize)
            .sorted { $0.offset < $1.offset }
        return best.map(\.text).joined(separator: " … ")
    }

    // MARK: Helpers

    private static func matches(_ pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { match in
            (1..<match.numberOfRanges).compactMap { Range(match.range(at: $0), in: text).map { String(text[$0]) } }
        }
    }

    private static func stripTags(_ html: String) -> String {
        let noTags = html.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        return decodeEntities(noTags)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func decodeEntities(_ text: String) -> String {
        var result = text
        let entities = ["&amp;": "&", "&quot;": "\"", "&#39;": "'", "&#x27;": "'", "&apos;": "'",
                        "&lt;": "<", "&gt;": ">", "&nbsp;": " ", "&#x2F;": "/", "&mdash;": "—", "&ndash;": "–"]
        for (entity, value) in entities {
            result = result.replacingOccurrences(of: entity, with: value)
        }
        return result
    }
}
