// ReleAF AI iOS SDK
// Production-ready Swift SDK for ReleAF AI Platform
// Version: 1.0.0

import Foundation
import UIKit

// MARK: - Configuration

public struct ReleAFConfig {
    public let baseURL: String
    public let apiKey: String?
    public let timeout: TimeInterval
    public let enableLogging: Bool
    public let enableCaching: Bool
    public let maxRetries: Int
    
    public init(
        baseURL: String,
        apiKey: String? = nil,
        timeout: TimeInterval = 30.0,
        enableLogging: Bool = false,
        enableCaching: Bool = true,
        maxRetries: Int = 3
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.timeout = timeout
        self.enableLogging = enableLogging
        self.enableCaching = enableCaching
        self.maxRetries = maxRetries
    }
    
    public static var production: ReleAFConfig {
        return ReleAFConfig(
            baseURL: "https://api.releaf.ai",
            timeout: 30.0,
            enableLogging: false,
            enableCaching: true,
            maxRetries: 3
        )
    }
    
    public static var development: ReleAFConfig {
        return ReleAFConfig(
            baseURL: "http://localhost:8080",
            timeout: 60.0,
            enableLogging: true,
            enableCaching: false,
            maxRetries: 1
        )
    }
}

// MARK: - Models

public struct ChatMessage: Codable {
    public let role: String
    public let content: String
    
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct Location: Codable {
    public let latitude: Double
    public let longitude: Double
    
    public init(latitude: Double, longitude: Double) {
        self.latitude = latitude
        self.longitude = longitude
    }
}

public struct ChatRequest: Codable {
    public let messages: [ChatMessage]
    public let location: Location?
    public let image: String?
    public let imageUrl: String?
    public let maxTokens: Int?
    public let temperature: Double?
    
    enum CodingKeys: String, CodingKey {
        case messages, location, image
        case imageUrl = "image_url"
        case maxTokens = "max_tokens"
        case temperature
    }
}

public struct ChatResponse: Codable {
    public let response: String
    public let sources: [Source]?
    public let suggestions: [String]?
    public let processingTimeMs: Double
    public let metadata: [String: AnyCodable]?
    
    enum CodingKeys: String, CodingKey {
        case response, sources, suggestions, metadata
        case processingTimeMs = "processing_time_ms"
    }
}

public struct Source: Codable {
    public let source: String
    public let docType: String?
    public let score: Double?
    
    enum CodingKeys: String, CodingKey {
        case source
        case docType = "doc_type"
        case score
    }
}

public struct VisionRequest: Codable {
    public let imageB64: String?
    public let imageUrl: String?
    public let enableDetection: Bool
    public let enableClassification: Bool
    public let enableRecommendations: Bool
    public let topK: Int
    
    enum CodingKeys: String, CodingKey {
        case imageB64 = "image_b64"
        case imageUrl = "image_url"
        case enableDetection = "enable_detection"
        case enableClassification = "enable_classification"
        case enableRecommendations = "enable_recommendations"
        case topK = "top_k"
    }
}

public struct VisionResponse: Codable {
    public let detections: [Detection]
    public let numDetections: Int
    public let classification: Classification?
    public let recommendations: [Recommendation]?
    public let imageSize: [Int]
    public let imageFormat: String
    public let imageQualityScore: Double
    public let confidenceScore: Double
    public let totalTimeMs: Double
    public let warnings: [String]
    public let errors: [String]
    
    enum CodingKeys: String, CodingKey {
        case detections, classification, recommendations, warnings, errors
        case numDetections = "num_detections"
        case imageSize = "image_size"
        case imageFormat = "image_format"
        case imageQualityScore = "image_quality_score"
        case confidenceScore = "confidence_score"
        case totalTimeMs = "total_time_ms"
    }
}

public struct Detection: Codable {
    public let bbox: [Double]
    public let className: String
    public let confidence: Double
    public let area: Double

    enum CodingKeys: String, CodingKey {
        case bbox, confidence, area
        case className = "class_name"
    }
}

public struct Classification: Codable {
    public let itemType: String
    public let itemConfidence: Double
    public let materialType: String
    public let materialConfidence: Double
    public let binType: String
    public let binConfidence: Double

    enum CodingKeys: String, CodingKey {
        case itemType = "item_type"
        case itemConfidence = "item_confidence"
        case materialType = "material_type"
        case materialConfidence = "material_confidence"
        case binType = "bin_type"
        case binConfidence = "bin_confidence"
    }
}

public struct Recommendation: Codable {
    public let targetMaterial: String
    public let score: Double
    public let difficulty: Int
    public let timeRequiredMinutes: Int
    public let toolsRequired: [String]
    public let skillsRequired: [String]

    enum CodingKeys: String, CodingKey {
        case score, difficulty
        case targetMaterial = "target_material"
        case timeRequiredMinutes = "time_required_minutes"
        case toolsRequired = "tools_required"
        case skillsRequired = "skills_required"
    }
}

public struct OrganizationSearchRequest: Codable {
    public let location: Location
    public let radiusKm: Double
    public let orgType: String?
    public let acceptedMaterials: [String]?
    public let limit: Int

    enum CodingKeys: String, CodingKey {
        case location, limit
        case radiusKm = "radius_km"
        case orgType = "org_type"
        case acceptedMaterials = "accepted_materials"
    }
}

public struct Organization: Codable {
    public let id: Int
    public let name: String
    public let orgType: String
    public let address: String
    public let city: String
    public let state: String
    public let zipCode: String
    public let latitude: Double
    public let longitude: Double
    public let distanceKm: Double
    public let phone: String?
    public let website: String?
    public let email: String?
    public let acceptedMaterials: [String]
    public let operatingHours: [String: String]?
    public let description: String?

    enum CodingKeys: String, CodingKey {
        case id, name, address, city, state, latitude, longitude, phone, website, email, description
        case orgType = "org_type"
        case zipCode = "zip_code"
        case distanceKm = "distance_km"
        case acceptedMaterials = "accepted_materials"
        case operatingHours = "operating_hours"
    }
}

public struct OrganizationSearchResponse: Codable {
    public let organizations: [Organization]
    public let numResults: Int
    public let searchLocation: Location
    public let searchRadiusKm: Double
    public let queryTimeMs: Double

    enum CodingKeys: String, CodingKey {
        case organizations
        case numResults = "num_results"
        case searchLocation = "search_location"
        case searchRadiusKm = "search_radius_km"
        case queryTimeMs = "query_time_ms"
    }
}

// Helper for dynamic JSON
public struct AnyCodable: Codable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intValue = try? container.decode(Int.self) {
            value = intValue
        } else if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
        } else if let stringValue = try? container.decode(String.self) {
            value = stringValue
        } else if let boolValue = try? container.decode(Bool.self) {
            value = boolValue
        } else if let arrayValue = try? container.decode([AnyCodable].self) {
            value = arrayValue.map { $0.value }
        } else if let dictValue = try? container.decode([String: AnyCodable].self) {
            value = dictValue.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let intValue as Int:
            try container.encode(intValue)
        case let doubleValue as Double:
            try container.encode(doubleValue)
        case let stringValue as String:
            try container.encode(stringValue)
        case let boolValue as Bool:
            try container.encode(boolValue)
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - Error Handling

public enum ReleAFError: Error {
    case invalidURL
    case networkError(Error)
    case invalidResponse
    case decodingError(Error)
    case serverError(Int, String)
    case rateLimitExceeded(retryAfter: Int)
    case unauthorized
    case timeout

    public var localizedDescription: String {
        switch self {
        case .invalidURL:
            return "Invalid API URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid server response"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let code, let message):
            return "Server error (\(code)): \(message)"
        case .rateLimitExceeded(let retryAfter):
            return "Rate limit exceeded. Retry after \(retryAfter) seconds"
        case .unauthorized:
            return "Unauthorized. Please check your API key"
        case .timeout:
            return "Request timeout"
        }
    }
}

// MARK: - Main SDK Client

public class ReleAFClient {
    private let config: ReleAFConfig
    private let session: URLSession
    private let cache: NSCache<NSString, CachedResponse>
    private let logger: Logger

    public init(config: ReleAFConfig) {
        self.config = config

        // Configure URLSession
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = config.timeout
        configuration.timeoutIntervalForResource = config.timeout * 2
        configuration.requestCachePolicy = config.enableCaching ? .returnCacheDataElseLoad : .reloadIgnoringLocalCacheData
        configuration.httpMaximumConnectionsPerHost = 5
        configuration.waitsForConnectivity = true

        self.session = URLSession(configuration: configuration)
        self.cache = NSCache<NSString, CachedResponse>()
        self.cache.countLimit = 100
        self.logger = Logger(enabled: config.enableLogging)
    }

    // MARK: - Chat API

    public func chat(
        messages: [ChatMessage],
        location: Location? = nil,
        image: UIImage? = nil,
        imageUrl: String? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        completion: @escaping (Result<ChatResponse, ReleAFError>) -> Void
    ) {
        logger.log("Chat request with \(messages.count) messages")

        var imageB64: String? = nil
        if let image = image {
            imageB64 = image.jpegData(compressionQuality: 0.8)?.base64EncodedString()
        }

        let request = ChatRequest(
            messages: messages,
            location: location,
            image: imageB64,
            imageUrl: imageUrl,
            maxTokens: maxTokens,
            temperature: temperature
        )

        performRequest(
            endpoint: "/api/v1/chat",
            method: "POST",
            body: request,
            completion: completion
        )
    }

    // MARK: - Vision API

    public func analyzeImage(
        image: UIImage? = nil,
        imageUrl: String? = nil,
        enableDetection: Bool = true,
        enableClassification: Bool = true,
        enableRecommendations: Bool = false,
        topK: Int = 5,
        completion: @escaping (Result<VisionResponse, ReleAFError>) -> Void
    ) {
        logger.log("Vision analysis request")

        var imageB64: String? = nil
        if let image = image {
            imageB64 = image.jpegData(compressionQuality: 0.8)?.base64EncodedString()
        }

        let request = VisionRequest(
            imageB64: imageB64,
            imageUrl: imageUrl,
            enableDetection: enableDetection,
            enableClassification: enableClassification,
            enableRecommendations: enableRecommendations,
            topK: topK
        )

        performRequest(
            endpoint: "/api/v1/vision/analyze",
            method: "POST",
            body: request,
            completion: completion
        )
    }

    // MARK: - Organization Search API

    public func searchOrganizations(
        location: Location,
        radiusKm: Double = 10.0,
        orgType: String? = nil,
        acceptedMaterials: [String]? = nil,
        limit: Int = 20,
        completion: @escaping (Result<OrganizationSearchResponse, ReleAFError>) -> Void
    ) {
        logger.log("Organization search request")

        let request = OrganizationSearchRequest(
            location: location,
            radiusKm: radiusKm,
            orgType: orgType,
            acceptedMaterials: acceptedMaterials,
            limit: limit
        )

        performRequest(
            endpoint: "/api/v1/organizations/search",
            method: "POST",
            body: request,
            completion: completion
        )
    }

