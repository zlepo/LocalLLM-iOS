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
        // Normal chat goes directly to the model.
        // Agent/tool routing remains available below for future MCP use.
        let context = buildContext(conversation)

        let directPrompt: String
        if context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            directPrompt = prompt
        } else {
            directPrompt = """
            CONVERSATION:
            \(context)

            CURRENT USER MESSAGE:
            \(prompt)
            """
        }

        return try await llm.generate(
            prompt: directPrompt,
            systemPrompt: systemPrompt,
            maxTokens: 1024
        )
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
    
    private func buildContext(_ conversation: [Message]) -> String {
        let recentMessages = conversation.suffix(4)
        return recentMessages.map { msg in
            let role = msg.role == .user ? "User" : "Assistant"
            return "\(role): \(msg.content)"
        }.joined(separator: "\n")
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
