# ReleAF AI - iOS API Documentation

**Version:** 1.0.0  
**Base URL:** `https://api.releaf.ai`  
**Protocol:** HTTPS  
**Format:** JSON

---

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [Endpoints](#endpoints)
   - [Chat API](#chat-api)
   - [Vision API](#vision-api)
   - [Organization Search API](#organization-search-api)
   - [Health Check](#health-check)
5. [Swift SDK Usage](#swift-sdk-usage)
6. [Response Formats](#response-formats)
7. [Best Practices](#best-practices)

---

## Authentication

### API Key Authentication

Include your API key in the request header:

```http
X-API-Key: your_api_key_here
```

### Swift Example

```swift
let config = ReleAFConfig(
    baseURL: "https://api.releaf.ai",
    apiKey: "your_api_key_here"
)
let client = ReleAFClient(config: config)
```

---

## Rate Limiting

**Default Limits:**
- 100 requests per minute per IP
- Burst size: 20 requests

**Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
Retry-After: 30
```

**429 Response:**
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 30,
  "limit": 100,
  "window": "1 minute"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "status_code": 400,
  "timestamp": "2025-12-15T10:30:00Z",
  "details": {}
}
```

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Verify API key |
| 429 | Rate Limit | Wait and retry |
| 500 | Server Error | Retry with exponential backoff |
| 503 | Service Unavailable | Service is down, retry later |
| 504 | Gateway Timeout | Request took too long, retry |

### Swift Error Handling

```swift
client.chat(messages: messages) { result in
    switch result {
    case .success(let response):
        print("Response: \(response.response)")
    case .failure(let error):
        switch error {
        case .rateLimitExceeded(let retryAfter):
            print("Rate limited. Retry after \(retryAfter)s")
        case .unauthorized:
            print("Invalid API key")
        case .networkError(let err):
            print("Network error: \(err)")
        default:
            print("Error: \(error.localizedDescription)")
        }
    }
}
```

---

## Endpoints

### Chat API

**Endpoint:** `POST /api/v1/chat`

**Description:** Multi-modal chat with intelligent orchestration. Supports text, images, and location context.

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "What can I do with this plastic bottle?"}
  ],
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "image": "base64_encoded_image_data",
  "max_tokens": 500,
  "temperature": 0.7
}
```

**Response:**

```json
{
  "response": "Here are some creative upcycling ideas for your plastic bottle...",
  "sources": [
    {
      "source": "Sustainability Guide 2024",
      "doc_type": "guide",
      "score": 0.95
    }
  ],
  "suggestions": [
    "Find recycling centers nearby",
    "Learn about plastic types"
  ],
  "processing_time_ms": 189.5,
  "metadata": {
    "services_used": ["vision", "llm", "rag"],
    "confidence": 0.92
  }
}
```

**Swift Example:**

```swift
let messages = [
    ChatMessage(role: "user", content: "How can I upcycle old jeans?")
]

client.chat(messages: messages) { result in
    switch result {
    case .success(let response):
        print(response.response)
        if let sources = response.sources {
            print("Sources: \(sources.count)")
        }
    case .failure(let error):
        print("Error: \(error)")
    }
}
```

**With Image:**

```swift
let image = UIImage(named: "waste_item")
let messages = [
    ChatMessage(role: "user", content: "What type of waste is this?")
]

client.chat(messages: messages, image: image) { result in
    // Handle response
}
```

---

### Vision API

**Endpoint:** `POST /api/v1/vision/analyze`

**Description:** Analyze images for waste recognition, classification, and upcycling recommendations.

**Request:**

```json
{
  "image_b64": "base64_encoded_image",
  "enable_detection": true,
  "enable_classification": true,
  "enable_recommendations": false,
  "top_k": 5
}
```

**Response:**

```json
{
  "detections": [
    {
      "bbox": [100, 150, 300, 400],
      "class_name": "plastic_bottle",
      "confidence": 0.95,
      "area": 40000
    }
  ],
  "num_detections": 1,
  "classification": {
    "item_type": "plastic_bottle",
    "item_confidence": 0.95,
    "material_type": "PET_plastic",
    "material_confidence": 0.92,
    "bin_type": "recycling",
    "bin_confidence": 0.98
  },
  "recommendations": null,
  "image_size": [640, 480],
  "image_format": "JPEG",
  "image_quality_score": 0.85,
  "confidence_score": 0.94,
  "total_time_ms": 145.2,
  "detection_time_ms": 85.3,
  "classification_time_ms": 59.9,
  "recommendation_time_ms": 0.0,
  "warnings": [],
  "errors": []
}
```

**Swift Example:**

```swift
let image = UIImage(named: "waste_item")!

client.analyzeImage(
    image: image,
    enableDetection: true,
    enableClassification: true,
    enableRecommendations: true,
    topK: 5
) { result in
    switch result {
    case .success(let response):
        print("Detected \(response.numDetections) items")

        if let classification = response.classification {
            print("Item: \(classification.itemType)")
            print("Material: \(classification.materialType)")
            print("Bin: \(classification.binType)")
        }

        if let recommendations = response.recommendations {
            for rec in recommendations {
                print("Upcycle to: \(rec.targetMaterial)")
                print("Difficulty: \(rec.difficulty)/5")
            }
        }

    case .failure(let error):
        print("Error: \(error)")
    }
}
```

---

### Organization Search API

**Endpoint:** `POST /api/v1/organizations/search`

**Description:** Find nearby charities, recycling centers, and sustainability organizations.

**Request:**

```json
{
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "radius_km": 10.0,
  "org_type": "recycling_center",
  "accepted_materials": ["plastic", "glass"],
  "limit": 20
}
```

**Response:**

```json
{
  "organizations": [
    {
      "id": 1,
      "name": "SF Recycling Center",
      "org_type": "recycling_center",
      "address": "123 Green St",
      "city": "San Francisco",
      "state": "CA",
      "zip_code": "94102",
      "latitude": 37.7750,
      "longitude": -122.4195,
      "distance_km": 0.5,
      "phone": "+1-415-555-0100",
      "website": "https://sfrecycling.org",
      "email": "info@sfrecycling.org",
      "accepted_materials": ["plastic", "glass", "metal", "paper"],
      "operating_hours": {
        "monday": "9:00-17:00",
        "tuesday": "9:00-17:00"
      },
      "description": "Full-service recycling facility"
    }
  ],
  "num_results": 1,
  "search_location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "search_radius_km": 10.0,
  "query_time_ms": 45.3
}
```

**Swift Example:**

```swift
import CoreLocation

// Get user location
let location = Location(
    latitude: 37.7749,
    longitude: -122.4194
)

client.searchOrganizations(
    location: location,
    radiusKm: 10.0,
    orgType: "recycling_center",
    acceptedMaterials: ["plastic", "glass"],
    limit: 20
) { result in
    switch result {
    case .success(let response):
        print("Found \(response.numResults) organizations")

        for org in response.organizations {
            print("\(org.name) - \(org.distanceKm) km away")
            print("Materials: \(org.acceptedMaterials.joined(separator: ", "))")
            if let website = org.website {
                print("Website: \(website)")
            }
        }

    case .failure(let error):
        print("Error: \(error)")
    }
}
```

---

### Health Check

**Endpoint:** `GET /health`

**Description:** Check API health and service status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:30:00Z",
  "services": {
    "orchestrator": {"healthy": true, "latency_ms": 5.2},
    "vision": {"healthy": true, "latency_ms": 8.1},
    "llm": {"healthy": true, "latency_ms": 12.3},
    "rag": {"healthy": true, "latency_ms": 6.5},
    "kg": {"healthy": true, "latency_ms": 7.8},
    "org_search": {"healthy": true, "latency_ms": 4.2}
  }
}
```

**Swift Example:**

```swift
client.healthCheck { result in
    switch result {
    case .success(let health):
        if let status = health["status"] as? String {
            print("API Status: \(status)")
        }
    case .failure(let error):
        print("Health check failed: \(error)")
    }
}
```

---

## Swift SDK Usage

### Installation

**Swift Package Manager:**

```swift
dependencies: [
    .package(url: "https://github.com/releaf-ai/ios-sdk.git", from: "1.0.0")
]
```

**CocoaPods:**

```ruby
pod 'ReleAFSDK', '~> 1.0.0'
```

### Initialization

```swift
import ReleAFSDK

// Production
let config = ReleAFConfig.production
config.apiKey = "your_api_key"
let client = ReleAFClient(config: config)

// Development
let devConfig = ReleAFConfig.development
let devClient = ReleAFClient(config: devConfig)
```

### Complete Example

```swift
import UIKit
import ReleAFSDK
import CoreLocation

class ViewController: UIViewController {
    let client = ReleAFClient(config: .production)

    func analyzeWaste(image: UIImage) {
        // Step 1: Analyze image
        client.analyzeImage(image: image) { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let visionResponse):
                // Step 2: Get upcycling ideas
                let itemType = visionResponse.classification?.itemType ?? "unknown"
                let messages = [
                    ChatMessage(
                        role: "user",
                        content: "How can I upcycle a \(itemType)?"
                    )
                ]

                self.client.chat(messages: messages) { chatResult in
                    switch chatResult {
                    case .success(let chatResponse):
                        DispatchQueue.main.async {
                            self.showResponse(chatResponse.response)
                        }
                    case .failure(let error):
                        print("Chat error: \(error)")
                    }
                }

            case .failure(let error):
                print("Vision error: \(error)")
            }
        }
    }

    func findNearbyOrganizations() {
        let location = Location(latitude: 37.7749, longitude: -122.4194)

        client.searchOrganizations(location: location) { result in
            switch result {
            case .success(let response):
                DispatchQueue.main.async {
                    self.showOrganizations(response.organizations)
                }
            case .failure(let error):
                print("Search error: \(error)")
            }
        }
    }
}
```

---

## Response Formats

### Formatted Answers

All chat responses include rich formatting:

- **Markdown**: For rich text display
- **HTML**: For web views
- **Plain Text**: For accessibility

**Example:**

```json
{
  "response": "## How to Upcycle Plastic Bottles\n\n1. Clean the bottle\n2. Cut carefully\n3. Sand edges",
  "metadata": {
    "answer_type": "how_to",
    "difficulty": "Easy",
    "time_estimate": "15 minutes"
  }
}
```

---

## Best Practices

### 1. Image Optimization

```swift
// Compress images before sending
extension UIImage {
    func optimizedForAPI() -> Data? {
        // Resize to max 1024x1024
        let maxSize: CGFloat = 1024
        let scale = min(maxSize / size.width, maxSize / size.height, 1.0)
        let newSize = CGSize(
            width: size.width * scale,
            height: size.height * scale
        )

        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        draw(in: CGRect(origin: .zero, size: newSize))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        // Compress to 80% quality
        return resized?.jpegData(compressionQuality: 0.8)
    }
}
```

### 2. Caching

```swift
// SDK handles caching automatically
let config = ReleAFConfig(
    baseURL: "https://api.releaf.ai",
    enableCaching: true  // Cache GET requests for 5 minutes
)
```

### 3. Error Handling

```swift
// Always handle all error cases
client.chat(messages: messages) { result in
    switch result {
    case .success(let response):
        // Handle success
        break
    case .failure(let error):
        switch error {
        case .rateLimitExceeded(let retryAfter):
            // Wait and retry
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(retryAfter)) {
                // Retry request
            }
        case .networkError:
            // Show offline message
            break
        case .timeout:
            // Show timeout message
            break
        default:
            // Show generic error
            break
        }
    }
}
```

### 4. Location Permissions

```swift
import CoreLocation

class LocationManager: NSObject, CLLocationManagerDelegate {
    let manager = CLLocationManager()

    func requestLocation() {
        manager.delegate = self
        manager.requestWhenInUseAuthorization()
        manager.requestLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.first else { return }

        let releafLocation = Location(
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude
        )

        // Use location in API calls
    }
}
```

### 5. Background Tasks

```swift
// Handle long-running requests
func performBackgroundRequest() {
    var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    backgroundTask = UIApplication.shared.beginBackgroundTask {
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }

    client.chat(messages: messages) { result in
        // Handle result

        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }
}
```

---

## Performance Metrics

**Expected Response Times:**

| Endpoint | Average | P95 | P99 |
|----------|---------|-----|-----|
| Chat (text only) | 150ms | 300ms | 500ms |
| Chat (with image) | 250ms | 500ms | 800ms |
| Vision Analysis | 150ms | 250ms | 400ms |
| Organization Search | 100ms | 200ms | 350ms |
| Health Check | 10ms | 20ms | 50ms |

**Throughput:**
- 100+ requests/second per service
- Auto-scaling to 1000+ concurrent users
- 99.9% uptime SLA

---

## Support

- **Documentation**: https://docs.releaf.ai
- **API Status**: https://status.releaf.ai
- **Support Email**: support@releaf.ai
- **GitHub**: https://github.com/releaf-ai/ios-sdk

---

**Last Updated:** 2025-12-15
**SDK Version:** 1.0.0
**API Version:** v1

