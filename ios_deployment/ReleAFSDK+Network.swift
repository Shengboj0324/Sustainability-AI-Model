// ReleAF AI iOS SDK - Network Layer
// Production-ready networking with retry logic, caching, and error handling

import Foundation

// MARK: - Cached Response

class CachedResponse {
    let data: Data
    let timestamp: Date
    let ttl: TimeInterval
    
    init(data: Data, ttl: TimeInterval = 300) {
        self.data = data
        self.timestamp = Date()
        self.ttl = ttl
    }
    
    var isValid: Bool {
        return Date().timeIntervalSince(timestamp) < ttl
    }
}

// MARK: - Logger

class Logger {
    private let enabled: Bool
    
    init(enabled: Bool) {
        self.enabled = enabled
    }
    
    func log(_ message: String) {
        if enabled {
            print("[ReleAF SDK] \(message)")
        }
    }
    
    func error(_ message: String) {
        if enabled {
            print("[ReleAF SDK ERROR] \(message)")
        }
    }
}

// MARK: - Network Extension

extension ReleAFClient {
    
    // Generic request performer with retry logic
    func performRequest<T: Codable, B: Codable>(
        endpoint: String,
        method: String,
        body: B?,
        retryCount: Int = 0,
        completion: @escaping (Result<T, ReleAFError>) -> Void
    ) {
        // Build URL
        guard let url = URL(string: config.baseURL + endpoint) else {
            completion(.failure(.invalidURL))
            return
        }
        
        // Check cache for GET requests
        if method == "GET", config.enableCaching {
            let cacheKey = url.absoluteString as NSString
            if let cached = cache.object(forKey: cacheKey), cached.isValid {
                logger.log("Cache hit for \(endpoint)")
                do {
                    let response = try JSONDecoder().decode(T.self, from: cached.data)
                    completion(.success(response))
                    return
                } catch {
                    logger.error("Cache decode error: \(error)")
                }
            }
        }
        
        // Build request
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("ReleAF-iOS-SDK/1.0.0", forHTTPHeaderField: "User-Agent")
        
        // Add API key if provided
        if let apiKey = config.apiKey {
            request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        }
        
        // Add body for POST/PUT requests
        if let body = body {
            do {
                request.httpBody = try JSONEncoder().encode(body)
            } catch {
                completion(.failure(.decodingError(error)))
                return
            }
        }
        
        // Perform request
        let task = session.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            
            // Handle network error
            if let error = error {
                if (error as NSError).code == NSURLErrorTimedOut {
                    completion(.failure(.timeout))
                } else if retryCount < self.config.maxRetries {
                    self.logger.log("Retrying request (\(retryCount + 1)/\(self.config.maxRetries))")
                    DispatchQueue.global().asyncAfter(deadline: .now() + Double(retryCount + 1)) {
                        self.performRequest(
                            endpoint: endpoint,
                            method: method,
                            body: body,
                            retryCount: retryCount + 1,
                            completion: completion
                        )
                    }
                } else {
                    completion(.failure(.networkError(error)))
                }
                return
            }
            
            // Handle HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(.invalidResponse))
                return
            }
            
            guard let data = data else {
                completion(.failure(.invalidResponse))
                return
            }
            
            // Handle status codes
            switch httpResponse.statusCode {
            case 200...299:
                // Success - decode response
                do {
                    let decoder = JSONDecoder()
                    let result = try decoder.decode(T.self, from: data)
                    
                    // Cache successful GET responses
                    if method == "GET", self.config.enableCaching {
                        let cacheKey = url.absoluteString as NSString
                        self.cache.setObject(CachedResponse(data: data), forKey: cacheKey)
                    }
                    
                    completion(.success(result))
                } catch {
                    self.logger.error("Decode error: \(error)")
                    completion(.failure(.decodingError(error)))
                }
                
            case 401:
                completion(.failure(.unauthorized))
                
            case 429:
                // Rate limit exceeded
                let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After") ?? "60"
                completion(.failure(.rateLimitExceeded(retryAfter: Int(retryAfter) ?? 60)))
                
            default:
                // Server error
                let message = String(data: data, encoding: .utf8) ?? "Unknown error"
                completion(.failure(.serverError(httpResponse.statusCode, message)))
            }
        }
        
        task.resume()
    }
}

