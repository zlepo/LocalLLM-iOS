import SwiftUI
import UniformTypeIdentifiers

@main
struct LocalLLMApp: App {
    @StateObject private var llmEngine = LLMEngine.shared
    @StateObject private var agentOrchestrator = AgentOrchestrator.shared
    @StateObject private var mcpManager = MCPManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(llmEngine)
                .environmentObject(agentOrchestrator)
                .environmentObject(mcpManager)
                .preferredColorScheme(.dark)
        }
    }
}

// MARK: - Main Content View

struct ContentView: View {
    @EnvironmentObject var llmEngine: LLMEngine
    @EnvironmentObject var orchestrator: AgentOrchestrator
    
    @State private var inputText: String = ""
    @State private var conversation: [Message] = []
    @State private var chatSessions: [ChatSession] = []
    @State private var activeChatID: UUID?
    @State private var isProcessing: Bool = false
    @State private var showSettings: Bool = false
    @State private var showMCPConfig: Bool = false
    @State private var showChatHistory: Bool = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                StatusBar(
                    modelLoaded: llmEngine.isModelLoaded,
                    activeAgents: orchestrator.activeAgentCount,
                    mcpConnections: MCPManager.shared.activeConnections
                )
                
                // Conversation view
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(conversation) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                            
                            if isProcessing {
                                ThinkingIndicator()
                            }
                        }
                        .padding()
                    }
                    .onChange(of: conversation.count) { _, _ in
                        saveConversation()

                        if let last = conversation.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // Input area
                InputBar(
                    text: $inputText,
                    isProcessing: isProcessing,
                    onSend: sendMessage
                )
            }
            .navigationTitle("Local LLM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showMCPConfig = true
                    } label: {
                        Image(systemName: "link.circle")
                    }
                }
                
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        showChatHistory = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }

                    Button {
                        startNewChat()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
            .sheet(isPresented: $showMCPConfig) {
                MCPConfigView()
            }
            .sheet(isPresented: $showChatHistory) {
                ChatHistoryView(
                    chats: chats,
                    activeChatID: activeChatID,
                    onSelect: { chatID in
                        selectChat(chatID)
                        showChatHistory = false
                    },
                    onNewChat: {
                        startNewChat()
                        showChatHistory = false
                    }
                )
            }
            .task {
                loadConversation()
                await loadModel()
            }
        }
    }
    

    private var applicationSupportDirectory: URL? {
        do {
            let directory = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )

            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )

            return directory
        } catch {
            print("Chat storage directory error: \(error.localizedDescription)")
            return nil
        }
    }

    private var chatsFileURL: URL? {
        applicationSupportDirectory?.appendingPathComponent("chats.json")
    }

    private var legacyConversationFileURL: URL? {
        applicationSupportDirectory?.appendingPathComponent("conversation.json")
    }

    private func loadConversation() {
        if let chatsURL = chatsFileURL,
           FileManager.default.fileExists(atPath: chatsURL.path) {
            do {
                let data = try Data(contentsOf: chatsURL)
                chatSessions = try JSONDecoder().decode([ChatSession].self, from: data)

                if let mostRecent = chatSessions.max(by: { $0.updatedAt < $1.updatedAt }) {
                    activeChatID = mostRecent.id
                    conversation = mostRecent.messages
                }

                print("Restored \(chatSessions.count) chat sessions")
                return
            } catch {
                print("Chat history restore failed: \(error.localizedDescription)")
            }
        }

        migrateLegacyConversationIfNeeded()
    }

    private func migrateLegacyConversationIfNeeded() {
        guard let legacyURL = legacyConversationFileURL,
              FileManager.default.fileExists(atPath: legacyURL.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: legacyURL)
            let legacyMessages = try JSONDecoder().decode([Message].self, from: data)

            guard !legacyMessages.isEmpty else { return }

            let firstUserText = legacyMessages.first(where: { $0.role == .user })?.content
            let migratedTitle = makeChatTitle(from: firstUserText ?? "Previous Chat")

            let migratedChat = ChatSession(
                title: migratedTitle,
                messages: legacyMessages
            )

            chatSessions = [migratedChat]
            activeChatID = migratedChat.id
            conversation = legacyMessages

            saveChats()

            print("Migrated legacy conversation with \(legacyMessages.count) messages")
        } catch {
            print("Legacy conversation migration failed: \(error.localizedDescription)")
        }
    }

    private func saveConversation() {
        guard !conversation.isEmpty else { return }

        if let activeChatID,
           let index = chatSessions.firstIndex(where: { $0.id == activeChatID }) {
            chatSessions[index].messages = conversation
            chatSessions[index].updatedAt = Date()

            if chatSessions[index].title == "New Chat",
               let firstUserText = conversation.first(where: { $0.role == .user })?.content {
                chatSessions[index].title = makeChatTitle(from: firstUserText)
            }
        } else {
            let title = makeChatTitle(
                from: conversation.first(where: { $0.role == .user })?.content ?? "New Chat"
            )

            let chat = ChatSession(
                title: title,
                messages: conversation
            )

            chatSessions.append(chat)
            activeChatID = chat.id
        }

        saveChats()
    }

    private func saveChats() {
        guard let url = chatsFileURL else { return }

        do {
            let data = try JSONEncoder().encode(chatSessions)
            try data.write(to: url, options: .atomic)
            print("Saved \(chatSessions.count) chat sessions")
        } catch {
            print("Chat history save failed: \(error.localizedDescription)")
        }
    }

    private func makeChatTitle(from text: String) -> String {
        let cleaned = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")

        guard !cleaned.isEmpty else { return "New Chat" }

        if cleaned.count <= 40 {
            return cleaned
        }

        return String(cleaned.prefix(40)) + "..."
    }

    private func startNewChat() {
        guard !isProcessing else { return }

        saveConversation()

        conversation = []
        activeChatID = nil
        inputText = ""
    }

    private func selectChat(_ chatID: UUID) {
        guard !isProcessing else { return }
        guard chatID != activeChatID else { return }
        guard let selectedChat = chatSessions.first(where: { $0.id == chatID }) else {
            print("Selected chat no longer exists")
            return
        }

        saveConversation()

        activeChatID = selectedChat.id
        conversation = selectedChat.messages
        inputText = ""
    }

    private func loadModel() async {
        await llmEngine.loadSavedModelIfAvailable()
    }
    
    private func sendMessage() {
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = Message(role: .user, content: inputText)
        conversation.append(userMessage)
        
        let prompt = inputText
        inputText = ""
        isProcessing = true
        
        Task {
            do {
                // Route through orchestrator - it decides if sub-agents are needed
                let response = try await orchestrator.process(
                    prompt: prompt,
                    conversation: Array(conversation.dropLast())
                )
                
                await MainActor.run {
                    conversation.append(Message(role: .assistant, content: response))
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    conversation.append(Message(
                        role: .system,
                        content: "Error: \(error.localizedDescription)"
                    ))
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct StatusBar: View {
    let modelLoaded: Bool
    let activeAgents: Int
    let mcpConnections: Int
    
    var body: some View {
        HStack {
            Label(
                modelLoaded ? "Model Ready" : "Loading...",
                systemImage: modelLoaded ? "checkmark.circle.fill" : "circle.dashed"
            )
            .foregroundStyle(modelLoaded ? .green : .orange)
            
            Spacer()
            
            if activeAgents > 0 {
                Label("\(activeAgents) agents", systemImage: "person.2.fill")
                    .foregroundStyle(.blue)
            }
            
            if mcpConnections > 0 {
                Label("\(mcpConnections) MCP", systemImage: "link")
                    .foregroundStyle(.purple)
            }
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.role == .user { Spacer() }
            
            VStack(alignment: message.role == .user ? .trailing : .leading) {
                if message.role == .system {
                    Text(message.content)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text(message.content)
                        .padding()
                        .background(
                            message.role == .user
                                ? Color.blue
                                : Color(.systemGray5)
                        )
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                if let agentInfo = message.agentInfo {
                    Text("via \(agentInfo)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: UIScreen.main.bounds.width * 0.75, alignment: message.role == .user ? .trailing : .leading)
            
            if message.role != .user { Spacer() }
        }
    }
}

struct InputBar: View {
    @Binding var text: String
    let isProcessing: Bool
    let onSend: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            TextField("Message", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color(.systemGray6))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .lineLimit(1...5)
            
            Button(action: onSend) {
                Image(systemName: isProcessing ? "stop.fill" : "arrow.up.circle.fill")
                    .font(.title)
                    .foregroundStyle(text.isEmpty ? .gray : .blue)
            }
            .disabled(text.isEmpty && !isProcessing)
        }
        .padding()
        .background(.ultraThinMaterial)
    }
}

struct ThinkingIndicator: View {
    @State private var dots = 0
    
    var body: some View {
        HStack {
            Text("Thinking" + String(repeating: ".", count: dots))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                dots = (dots + 1) % 4
            }
        }
    }
}

struct ChatHistoryView: View {
    let chats: [ChatSession]
    let activeChatID: UUID?
    let onSelect: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss

    private var sortedChats: [ChatSession] {
        chats.sorted { $0.updatedAt > $1.updatedAt }
    }

    var body: some View {
        NavigationStack {
            Group {
                if sortedChats.isEmpty {
                    ContentUnavailableView(
                        "No Chat History",
                        systemImage: "bubble.left.and.bubble.right",
                        description: Text("Your saved conversations will appear here.")
                    )
                } else {
                    List(sortedChats) { chat in
                        Button {
                            onSelect(chat.id)
                            dismiss()
                        } label: {
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(chat.title)
                                        .font(.headline)
                                        .foregroundStyle(.primary)
                                        .lineLimit(2)

                                    HStack(spacing: 6) {
                                        Text(chat.updatedAt, style: .date)

                                        Text("•")

                                        Text("\(chat.messages.count) messages")
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }

                                Spacer()

                                if chat.id == activeChatID {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.blue)
                                } else {
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .navigationTitle("Chat History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
// MARK: - Data Models

struct Message: Identifiable, Codable {
    let id: UUID
    let role: Role
    let content: String
    var agentInfo: String?
    let timestamp: Date

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        agentInfo: String? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.agentInfo = agentInfo
        self.timestamp = timestamp
    }

    enum Role: String, Codable {
        case user
        case assistant
        case system
    }
}

struct ChatSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [Message]
    let createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        title: String = "New Chat",
        messages: [Message] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
// MARK: - Placeholder Views

struct SettingsView: View {
    @EnvironmentObject var llmEngine: LLMEngine
    @State private var showModelPicker = false

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    Button("Choose GGUF Model") {
                        showModelPicker = true
                    }
                    .disabled(llmEngine.isGenerating)

                    Text(llmEngine.modelStatus)
                        .foregroundStyle(llmEngine.isModelLoaded ? .green : .secondary)

                    if let error = llmEngine.lastError {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    if let modelName = llmEngine.loadedModelName {
                        Text(modelName)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Text("Memory: ~4.5GB")
                }

                Section("Performance") {
                    Text("Metal GPU: Enabled")
                    Text("Threads: 6")
                }
            }
            .fileImporter(
                isPresented: $showModelPicker,
                allowedContentTypes: [.data]
            ) { result in
                switch result {
                case .success(let url):
                    Task {
                        do {
                            try await llmEngine.importModel(from: url)
                        } catch {
                            print("GGUF import failed: \(error.localizedDescription)")
                        }
                    }

                case .failure(let error):
                    print("GGUF picker failed: \(error.localizedDescription)")
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct MCPConfigView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Connected Servers") {
                    Text("No servers connected")
                        .foregroundStyle(.secondary)
                }
                
                Section {
                    Button("Add MCP Server") {}
                }
            }
            .navigationTitle("MCP Connections")
        }
    }
}
