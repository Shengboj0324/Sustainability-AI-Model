# ReleAF AI - iOS Frontend Integration Guide

**Version:** 1.0.0  
**Target:** iOS 14.0+  
**Framework:** SwiftUI / UIKit

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [UI Components](#ui-components)
3. [Chat Interface](#chat-interface)
4. [Image Analysis Interface](#image-analysis-interface)
5. [Organization Search Interface](#organization-search-interface)
6. [State Management](#state-management)
7. [Performance Optimization](#performance-optimization)
8. [Offline Support](#offline-support)
9. [Accessibility](#accessibility)
10. [Testing](#testing)

---

## Architecture Overview

### Recommended Architecture: MVVM + Coordinator

```
┌─────────────────────────────────────────────────────────┐
│                      App Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Coordinator  │  │   AppState   │  │  ReleAFSDK   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Feature Modules                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Chat Module  │  │Vision Module │  │  Org Module  │ │
│  │              │  │              │  │              │ │
│  │ View         │  │ View         │  │ View         │ │
│  │ ViewModel    │  │ ViewModel    │  │ ViewModel    │ │
│  │ Model        │  │ Model        │  │ Model        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                   Infrastructure                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Network    │  │   Storage    │  │   Location   │ │
│  │   Manager    │  │   Manager    │  │   Manager    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ReleAFApp/
├── App/
│   ├── AppDelegate.swift
│   ├── SceneDelegate.swift
│   └── AppCoordinator.swift
├── Features/
│   ├── Chat/
│   │   ├── Views/
│   │   │   ├── ChatView.swift
│   │   │   ├── MessageBubbleView.swift
│   │   │   ├── InputBarView.swift
│   │   │   └── TypingIndicatorView.swift
│   │   ├── ViewModels/
│   │   │   └── ChatViewModel.swift
│   │   └── Models/
│   │       └── ChatModels.swift
│   ├── Vision/
│   │   ├── Views/
│   │   │   ├── VisionAnalysisView.swift
│   │   │   ├── ImagePickerView.swift
│   │   │   └── ResultsView.swift
│   │   ├── ViewModels/
│   │   │   └── VisionViewModel.swift
│   │   └── Models/
│   │       └── VisionModels.swift
│   └── Organizations/
│       ├── Views/
│       │   ├── OrganizationListView.swift
│       │   ├── OrganizationMapView.swift
│       │   └── OrganizationDetailView.swift
│       ├── ViewModels/
│       │   └── OrganizationViewModel.swift
│       └── Models/
│           └── OrganizationModels.swift
├── Infrastructure/
│   ├── Network/
│   │   ├── NetworkManager.swift
│   │   └── APIClient.swift
│   ├── Storage/
│   │   ├── CacheManager.swift
│   │   └── PersistenceManager.swift
│   └── Location/
│       └── LocationManager.swift
├── Common/
│   ├── Extensions/
│   ├── Utilities/
│   └── Constants/
└── Resources/
    ├── Assets.xcassets
    ├── Localizable.strings
    └── Info.plist
```

---

## UI Components

### 1. Chat Interface Components

#### MessageBubbleView.swift (SwiftUI)

```swift
import SwiftUI

struct MessageBubbleView: View {
    let message: ChatMessage
    let isUser: Bool
    
    var body: some View {
        HStack {
            if isUser { Spacer() }
            
            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding(12)
                    .background(isUser ? Color.blue : Color.gray.opacity(0.2))
                    .foregroundColor(isUser ? .white : .primary)
                    .cornerRadius(16)
                    .textSelection(.enabled)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !isUser { Spacer() }
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
    }
}
```

#### InputBarView.swift (SwiftUI)

```swift
import SwiftUI

struct InputBarView: View {
    @Binding var text: String
    @Binding var isLoading: Bool
    let onSend: () -> Void
    let onImagePicker: () -> Void
    let onLocationToggle: () -> Void
    
    @State private var isLocationEnabled = false
    
    var body: some View {
        HStack(spacing: 12) {
            // Location button
            Button(action: {
                isLocationEnabled.toggle()
                onLocationToggle()
            }) {
                Image(systemName: isLocationEnabled ? "location.fill" : "location")
                    .foregroundColor(isLocationEnabled ? .blue : .gray)
            }
            
            // Image picker button
            Button(action: onImagePicker) {
                Image(systemName: "photo")
                    .foregroundColor(.gray)
            }
            
            // Text input
            TextField("Ask about sustainability...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .disabled(isLoading)
            
            // Send button
            Button(action: onSend) {
                if isLoading {
                    ProgressView()
                        .frame(width: 24, height: 24)
                } else {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 28))
                        .foregroundColor(text.isEmpty ? .gray : .blue)
                }
            }
            .disabled(text.isEmpty || isLoading)
        }
        .padding()
        .background(Color(.systemBackground))
    }
}
```

#### TypingIndicatorView.swift (SwiftUI)

```swift
import SwiftUI

struct TypingIndicatorView: View {
    @State private var animationPhase = 0
    
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { index in
                Circle()
                    .fill(Color.gray)
                    .frame(width: 8, height: 8)
                    .scaleEffect(animationPhase == index ? 1.2 : 1.0)
                    .animation(
                        Animation.easeInOut(duration: 0.6)
                            .repeatForever()
                            .delay(Double(index) * 0.2),
                        value: animationPhase
                    )
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.2))
        .cornerRadius(16)
        .onAppear {
            animationPhase = 0
        }
    }
}
```

---

### 2. Complete Chat View Implementation

#### ChatViewModel.swift

```swift
import Foundation
import Combine
import ReleAFSDK

class ChatViewModel: ObservableObject {
    @Published var messages: [DisplayMessage] = []
    @Published var inputText = ""
    @Published var isLoading = false
    @Published var error: String?
    @Published var isLocationEnabled = false

    private let client: ReleAFClient
    private var cancellables = Set<AnyCancellable>()
    private var currentLocation: Location?

    init(client: ReleAFClient) {
        self.client = client
    }

    func sendMessage(image: UIImage? = nil) {
        guard !inputText.isEmpty || image != nil else { return }

        // Add user message
        let userMessage = DisplayMessage(
            id: UUID(),
            role: "user",
            content: inputText,
            timestamp: Date(),
            image: image
        )
        messages.append(userMessage)

        let messageText = inputText
        inputText = ""
        isLoading = true
        error = nil

        // Prepare chat messages
        let chatMessages = messages.map { msg in
            ChatMessage(role: msg.role, content: msg.content)
        }

        // Send to API
        client.chat(
            messages: chatMessages,
            location: isLocationEnabled ? currentLocation : nil,
            image: image
        ) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoading = false

                switch result {
                case .success(let response):
                    let assistantMessage = DisplayMessage(
                        id: UUID(),
                        role: "assistant",
                        content: response.response,
                        timestamp: Date(),
                        sources: response.sources,
                        suggestions: response.suggestions
                    )
                    self?.messages.append(assistantMessage)

                case .failure(let error):
                    self?.error = error.localizedDescription
                    // Remove user message on error
                    self?.messages.removeAll { $0.id == userMessage.id }
                }
            }
        }
    }

    func updateLocation(_ location: Location) {
        self.currentLocation = location
    }

    func clearChat() {
        messages.removeAll()
        error = nil
    }
}

struct DisplayMessage: Identifiable {
    let id: UUID
    let role: String
    let content: String
    let timestamp: Date
    var image: UIImage?
    var sources: [Source]?
    var suggestions: [String]?
}
```

#### ChatView.swift (SwiftUI)

```swift
import SwiftUI
import ReleAFSDK

struct ChatView: View {
    @StateObject private var viewModel: ChatViewModel
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?

    init(client: ReleAFClient) {
        _viewModel = StateObject(wrappedValue: ChatViewModel(client: client))
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView

            Divider()

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(viewModel.messages) { message in
                            MessageBubbleView(
                                message: message,
                                isUser: message.role == "user"
                            )
                            .id(message.id)
                        }

                        if viewModel.isLoading {
                            HStack {
                                TypingIndicatorView()
                                Spacer()
                            }
                            .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                }
                .onChange(of: viewModel.messages.count) { _ in
                    if let lastMessage = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Error banner
            if let error = viewModel.error {
                ErrorBannerView(message: error) {
                    viewModel.error = nil
                }
            }

            Divider()

            // Input bar
            InputBarView(
                text: $viewModel.inputText,
                isLoading: $viewModel.isLoading,
                onSend: {
                    if let image = selectedImage {
                        viewModel.sendMessage(image: image)
                        selectedImage = nil
                    } else {
                        viewModel.sendMessage()
                    }
                },
                onImagePicker: {
                    showImagePicker = true
                },
                onLocationToggle: {
                    viewModel.isLocationEnabled.toggle()
                }
            )
        }
        .navigationTitle("ReleAF AI")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $selectedImage)
        }
    }

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("ReleAF AI")
                    .font(.headline)
                Text("Sustainability Assistant")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: viewModel.clearChat) {
                Image(systemName: "trash")
                    .foregroundColor(.red)
            }
        }
        .padding()
    }
}
```

---

### 3. Vision Analysis Interface

#### VisionViewModel.swift

```swift
import Foundation
import Combine
import ReleAFSDK
import UIKit

class VisionViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var analysisResult: VisionResponse?
    @Published var isAnalyzing = false
    @Published var error: String?

    private let client: ReleAFClient

    init(client: ReleAFClient) {
        self.client = client
    }

    func analyzeImage() {
        guard let image = selectedImage else { return }

        isAnalyzing = true
        error = nil

        client.analyzeImage(
            image: image,
            enableDetection: true,
            enableClassification: true,
            enableRecommendations: true,
            topK: 5
        ) { [weak self] result in
            DispatchQueue.main.async {
                self?.isAnalyzing = false

                switch result {
                case .success(let response):
                    self?.analysisResult = response
                case .failure(let error):
                    self?.error = error.localizedDescription
                }
            }
        }
    }

    func reset() {
        selectedImage = nil
        analysisResult = nil
        error = nil
    }
}
```

#### VisionAnalysisView.swift (SwiftUI)

```swift
import SwiftUI
import ReleAFSDK

struct VisionAnalysisView: View {
    @StateObject private var viewModel: VisionViewModel
    @State private var showImagePicker = false

    init(client: ReleAFClient) {
        _viewModel = StateObject(wrappedValue: VisionViewModel(client: client))
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image selection
                if let image = viewModel.selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    Button(action: { showImagePicker = true }) {
                        VStack(spacing: 12) {
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 60))
                                .foregroundColor(.blue)
                            Text("Select Image to Analyze")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 200)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                    }
                }

                // Action buttons
                if viewModel.selectedImage != nil {
                    HStack(spacing: 12) {
                        Button(action: { showImagePicker = true }) {
                            Label("Change", systemImage: "photo")
                        }
                        .buttonStyle(.bordered)

                        Button(action: viewModel.analyzeImage) {
                            if viewModel.isAnalyzing {
                                ProgressView()
                            } else {
                                Label("Analyze", systemImage: "sparkles")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isAnalyzing)
                    }
                }

                // Results
                if let result = viewModel.analysisResult {
                    VStack(alignment: .leading, spacing: 16) {
                        // Classification
                        if let classification = result.classification {
                            ClassificationResultView(classification: classification)
                        }

                        // Detections
                        if !result.detections.isEmpty {
                            DetectionResultsView(detections: result.detections)
                        }

                        // Recommendations
                        if let recommendations = result.recommendations, !recommendations.isEmpty {
                            RecommendationsView(recommendations: recommendations)
                        }
                    }
                }

                // Error
                if let error = viewModel.error {
                    ErrorBannerView(message: error) {
                        viewModel.error = nil
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Waste Analysis")
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $viewModel.selectedImage)
        }
    }
}

struct ClassificationResultView: View {
    let classification: Classification

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Classification")
                .font(.headline)

            ResultRow(
                label: "Item Type",
                value: classification.itemType.replacingOccurrences(of: "_", with: " ").capitalized,
                confidence: classification.itemConfidence
            )

            ResultRow(
                label: "Material",
                value: classification.materialType.replacingOccurrences(of: "_", with: " ").capitalized,
                confidence: classification.materialConfidence
            )

            ResultRow(
                label: "Disposal Bin",
                value: classification.binType.capitalized,
                confidence: classification.binConfidence,
                color: binColor(for: classification.binType)
            )
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }

    func binColor(for binType: String) -> Color {
        switch binType.lowercased() {
        case "recycling": return .blue
        case "compost": return .green
        case "landfill": return .gray
        default: return .primary
        }
    }
}

struct ResultRow: View {
    let label: String
    let value: String
    let confidence: Double
    var color: Color = .primary

    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(value)
                    .fontWeight(.semibold)
                    .foregroundColor(color)
                Text("\(Int(confidence * 100))% confident")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}
```


