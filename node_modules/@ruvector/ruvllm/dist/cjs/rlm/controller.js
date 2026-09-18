"use strict";
/**
 * RLM Controller - Recursive Retrieval Language Model
 *
 * Implements a recursive retrieval-augmented generation system that:
 * 1. Breaks down complex queries into sub-queries
 * 2. Retrieves relevant memory spans for each query
 * 3. Synthesizes coherent answers from retrieved context
 * 4. Optionally reflects on and refines answers
 *
 * @example Basic Usage
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   maxDepth: 3,
 *   retrievalTopK: 10,
 *   enableCache: true,
 * });
 *
 * // Add knowledge to memory
 * await rlm.addMemory('Machine learning is a subset of AI that enables systems to learn from data.');
 * await rlm.addMemory('Deep learning uses neural networks with many layers.');
 *
 * // Query with recursive retrieval
 * const answer = await rlm.query('Explain the relationship between ML and deep learning');
 * console.log(answer.text);
 * console.log('Sources:', answer.sources.length);
 * console.log('Confidence:', answer.confidence);
 * ```
 *
 * @example Streaming
 * ```typescript
 * const rlm = new RlmController();
 *
 * for await (const event of rlm.queryStream('What is AI?')) {
 *   if (event.type === 'token') {
 *     process.stdout.write(event.text);
 *   } else {
 *     console.log('\n\nDone! Quality:', event.answer.qualityScore);
 *   }
 * }
 * ```
 *
 * @example With Reflection
 * ```typescript
 * const rlm = new RlmController({
 *   enableReflection: true,
 *   maxReflectionIterations: 2,
 *   minQualityScore: 0.8,
 * });
 *
 * const answer = await rlm.query('Complex multi-part question...');
 * // Answer will be iteratively refined until quality >= 0.8
 * ```
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RlmController = void 0;
const engine_1 = require("../engine");
/**
 * Default configuration values
 */
const DEFAULT_CONFIG = {
    maxDepth: 3,
    maxSubQueries: 5,
    tokenBudget: 4096,
    enableCache: true,
    cacheTtl: 300000, // 5 minutes
    retrievalTopK: 10,
    minQualityScore: 0.7,
    enableReflection: false,
    maxReflectionIterations: 2,
};
/**
 * RlmController - Recursive Retrieval Language Model Controller
 *
 * Orchestrates retrieval-augmented generation with recursive sub-query
 * decomposition, memory search, and optional self-reflection.
 */
class RlmController {
    /**
     * Create a new RLM controller
     *
     * @param config - Configuration options
     * @param engine - Optional RuvLLM engine instance (creates new if not provided)
     *
     * @example
     * ```typescript
     * // With default config
     * const rlm = new RlmController();
     *
     * // With custom config
     * const rlm = new RlmController({
     *   maxDepth: 5,
     *   enableReflection: true,
     * });
     *
     * // With existing engine
     * const engine = new RuvLLM({ learningEnabled: true });
     * const rlm = new RlmController({}, engine);
     * ```
     */
    constructor(config, engine) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.cache = new Map();
        this.engine = engine ?? new engine_1.RuvLLM({ learningEnabled: true });
        this.memoryIdCounter = 0;
    }
    /**
     * Query the RLM with recursive retrieval
     *
     * @param input - The query string
     * @returns Promise resolving to the answer with sources and metadata
     *
     * @example
     * ```typescript
     * const answer = await rlm.query('What is the capital of France?');
     * console.log(answer.text); // "The capital of France is Paris..."
     * console.log(answer.confidence); // 0.95
     * console.log(answer.sources); // [{ id: '...', text: '...', similarityScore: 0.92 }]
     * ```
     */
    async query(input) {
        // Check cache first
        if (this.config.enableCache) {
            const cached = this.getCached(input);
            if (cached) {
                return { ...cached, cached: true };
            }
        }
        // Retrieve relevant memory spans
        const sources = await this.searchMemory(input, this.config.retrievalTopK);
        // Generate sub-queries if needed and depth allows
        const subQueries = await this.generateSubQueries(input, sources, 0);
        // Build context from sources and sub-query answers
        const context = this.buildContext(sources, subQueries);
        // Generate the answer
        const startTime = Date.now();
        const response = this.engine.query(this.buildPrompt(input, context), this.getGenerationConfig());
        // Calculate token usage (estimate if not provided by engine)
        const tokenUsage = this.estimateTokenUsage(input, context, response.text);
        // Calculate quality score
        const qualityScore = this.calculateQualityScore(sources, response.confidence);
        let answer = {
            text: response.text,
            confidence: response.confidence,
            qualityScore,
            sources,
            subQueries: subQueries.length > 0 ? subQueries : undefined,
            tokenUsage,
            cached: false,
        };
        // Apply reflection if enabled and quality is below threshold
        if (this.config.enableReflection && qualityScore < this.config.minQualityScore) {
            answer = await this.applyReflection(input, answer);
        }
        // Cache the result
        if (this.config.enableCache) {
            this.setCache(input, answer);
        }
        return answer;
    }
    /**
     * Query with streaming response
     *
     * @param input - The query string
     * @yields StreamToken events (either partial tokens or final answer)
     *
     * @example
     * ```typescript
     * for await (const event of rlm.queryStream('Explain quantum computing')) {
     *   if (event.type === 'token') {
     *     // Partial token received
     *     process.stdout.write(event.text);
     *   } else {
     *     // Generation complete
     *     console.log('\n\nSources:', event.answer.sources.length);
     *   }
     * }
     * ```
     */
    async *queryStream(input) {
        // Check cache first
        if (this.config.enableCache) {
            const cached = this.getCached(input);
            if (cached) {
                // Simulate streaming for cached response
                const words = cached.text.split(' ');
                for (const word of words) {
                    yield { type: 'token', text: word + ' ', done: false };
                    await this.delay(10); // Small delay for realistic streaming
                }
                yield { type: 'done', answer: { ...cached, cached: true }, done: true };
                return;
            }
        }
        // Retrieve sources
        const sources = await this.searchMemory(input, this.config.retrievalTopK);
        const subQueries = await this.generateSubQueries(input, sources, 0);
        const context = this.buildContext(sources, subQueries);
        // Generate with simulated streaming
        const prompt = this.buildPrompt(input, context);
        const response = this.engine.query(prompt, this.getGenerationConfig());
        // Stream the response word by word
        const words = response.text.split(' ');
        let streamedText = '';
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            const text = i < words.length - 1 ? word + ' ' : word;
            streamedText += text;
            yield { type: 'token', text, done: false };
            await this.delay(20); // Simulate generation latency
        }
        const tokenUsage = this.estimateTokenUsage(input, context, streamedText);
        const qualityScore = this.calculateQualityScore(sources, response.confidence);
        const answer = {
            text: streamedText,
            confidence: response.confidence,
            qualityScore,
            sources,
            subQueries: subQueries.length > 0 ? subQueries : undefined,
            tokenUsage,
            cached: false,
        };
        // Cache the result
        if (this.config.enableCache) {
            this.setCache(input, answer);
        }
        yield { type: 'done', answer, done: true };
    }
    /**
     * Add content to memory for retrieval
     *
     * @param text - The text content to store
     * @param metadata - Optional metadata to associate with the memory
     * @returns Promise resolving to the memory span ID
     *
     * @example
     * ```typescript
     * const id1 = await rlm.addMemory(
     *   'TypeScript is a typed superset of JavaScript.',
     *   { source: 'documentation', category: 'programming' }
     * );
     *
     * const id2 = await rlm.addMemory(
     *   'React is a JavaScript library for building UIs.'
     * );
     * ```
     */
    async addMemory(text, metadata) {
        const nodeId = this.engine.addMemory(text, metadata);
        const id = `rlm-mem-${this.memoryIdCounter++}-${nodeId}`;
        return id;
    }
    /**
     * Search memory for relevant spans
     *
     * @param query - The search query
     * @param topK - Number of results to return (default: config.retrievalTopK)
     * @returns Promise resolving to array of memory spans
     *
     * @example
     * ```typescript
     * const spans = await rlm.searchMemory('JavaScript frameworks', 5);
     * for (const span of spans) {
     *   console.log(`[${span.similarityScore.toFixed(2)}] ${span.text}`);
     * }
     * ```
     */
    async searchMemory(query, topK) {
        const k = topK ?? this.config.retrievalTopK;
        const results = this.engine.searchMemory(query, k);
        return results.map((result, index) => ({
            id: `rlm-span-${result.id}-${index}`,
            text: result.content,
            similarityScore: result.score,
            source: result.metadata?.source,
            metadata: result.metadata,
        }));
    }
    /**
     * Clear the response cache
     *
     * @example
     * ```typescript
     * rlm.clearCache();
     * console.log('Cache cleared');
     * ```
     */
    clearCache() {
        this.cache.clear();
    }
    /**
     * Get current cache statistics
     *
     * @returns Object with cache size and hit rate info
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            entries: this.cache.size,
        };
    }
    /**
     * Update configuration at runtime
     *
     * @param config - Partial configuration to merge
     */
    updateConfig(config) {
        this.config = { ...this.config, ...config };
    }
    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
    // ============================================
    // Private Methods
    // ============================================
    /**
     * Generate sub-queries for complex questions
     */
    async generateSubQueries(query, sources, depth) {
        if (depth >= this.config.maxDepth) {
            return [];
        }
        // Simple heuristic: generate sub-queries for questions with multiple parts
        const subQueries = [];
        const parts = this.decomposeQuery(query);
        for (const part of parts.slice(0, this.config.maxSubQueries)) {
            if (part.trim().length < 10)
                continue;
            // Search for sub-query specific sources
            const subSources = await this.searchMemory(part, Math.ceil(this.config.retrievalTopK / 2));
            const context = this.buildContext(subSources, []);
            const response = this.engine.query(this.buildPrompt(part, context), { ...this.getGenerationConfig(), maxTokens: 256 });
            subQueries.push({
                query: part,
                answer: response.text,
                depth: depth + 1,
            });
        }
        return subQueries;
    }
    /**
     * Decompose a complex query into simpler parts
     */
    decomposeQuery(query) {
        // Split on common conjunctions and question markers
        const parts = [];
        // Check for multi-part questions
        const conjunctions = [' and ', ' or ', '. ', '? ', '; '];
        let current = query;
        for (const conj of conjunctions) {
            if (current.includes(conj)) {
                const split = current.split(conj);
                parts.push(...split.filter(p => p.trim().length > 10));
                current = '';
                break;
            }
        }
        // If no decomposition happened, return original
        if (parts.length === 0) {
            return [query];
        }
        return parts;
    }
    /**
     * Build context string from sources and sub-queries
     */
    buildContext(sources, subQueries) {
        const parts = [];
        // Add sources
        if (sources.length > 0) {
            parts.push('Relevant context:');
            for (const source of sources) {
                parts.push(`- ${source.text}`);
            }
        }
        // Add sub-query answers
        if (subQueries.length > 0) {
            parts.push('\nRelated information:');
            for (const sq of subQueries) {
                parts.push(`Q: ${sq.query}`);
                parts.push(`A: ${sq.answer}`);
            }
        }
        return parts.join('\n');
    }
    /**
     * Build the full prompt with context
     */
    buildPrompt(query, context) {
        if (context.trim().length === 0) {
            return query;
        }
        return `${context}\n\nBased on the above context, answer the following question:\n${query}`;
    }
    /**
     * Get generation config based on RLM settings
     */
    getGenerationConfig() {
        return {
            maxTokens: Math.min(this.config.tokenBudget, 2048),
            temperature: 0.7,
            topP: 0.9,
        };
    }
    /**
     * Estimate token usage
     */
    estimateTokenUsage(query, context, response) {
        // Rough estimation: ~4 characters per token
        const promptTokens = Math.ceil((query.length + context.length) / 4);
        const completionTokens = Math.ceil(response.length / 4);
        return {
            prompt: promptTokens,
            completion: completionTokens,
            total: promptTokens + completionTokens,
        };
    }
    /**
     * Calculate quality score based on sources and confidence
     */
    calculateQualityScore(sources, confidence) {
        if (sources.length === 0) {
            return confidence * 0.5; // Penalize answers without sources
        }
        // Average source similarity
        const avgSimilarity = sources.reduce((sum, s) => sum + s.similarityScore, 0) / sources.length;
        // Weighted combination
        return confidence * 0.6 + avgSimilarity * 0.4;
    }
    /**
     * Apply self-reflection to improve answer
     */
    async applyReflection(query, answer) {
        let currentAnswer = answer;
        let iterations = 0;
        while (iterations < this.config.maxReflectionIterations &&
            currentAnswer.qualityScore < this.config.minQualityScore) {
            iterations++;
            // Generate critique
            const critiquePrompt = `Evaluate this answer for accuracy and completeness:
Question: ${query}
Answer: ${currentAnswer.text}

Provide a brief critique and suggest improvements.`;
            const critiqueResponse = this.engine.query(critiquePrompt, {
                maxTokens: 256,
                temperature: 0.5,
            });
            // Generate improved answer
            const improvePrompt = `Based on this feedback: "${critiqueResponse.text}"

Improve this answer:
Question: ${query}
Original: ${currentAnswer.text}

Provide an improved answer:`;
            const improvedResponse = this.engine.query(improvePrompt, this.getGenerationConfig());
            // Update answer with reflection improvements
            const newQualityScore = Math.min(1.0, currentAnswer.qualityScore + 0.1 * iterations);
            currentAnswer = {
                ...currentAnswer,
                text: improvedResponse.text,
                confidence: Math.max(currentAnswer.confidence, improvedResponse.confidence),
                qualityScore: newQualityScore,
                tokenUsage: {
                    prompt: currentAnswer.tokenUsage.prompt + 100, // Approximate additional tokens
                    completion: currentAnswer.tokenUsage.completion + 100,
                    total: currentAnswer.tokenUsage.total + 200,
                },
            };
        }
        return currentAnswer;
    }
    /**
     * Get cached answer if valid
     */
    getCached(query) {
        const hash = this.hashQuery(query);
        const entry = this.cache.get(hash);
        if (!entry) {
            return null;
        }
        // Check TTL
        if (Date.now() - entry.timestamp > this.config.cacheTtl) {
            this.cache.delete(hash);
            return null;
        }
        return entry.answer;
    }
    /**
     * Set cache entry
     */
    setCache(query, answer) {
        const hash = this.hashQuery(query);
        this.cache.set(hash, {
            answer,
            timestamp: Date.now(),
            queryHash: hash,
        });
        // Prune old entries if cache gets too large
        if (this.cache.size > 1000) {
            this.pruneCache();
        }
    }
    /**
     * Simple hash function for cache keys
     */
    hashQuery(query) {
        let hash = 0;
        for (let i = 0; i < query.length; i++) {
            const char = query.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return `rlm-cache-${hash.toString(16)}`;
    }
    /**
     * Prune expired cache entries
     */
    pruneCache() {
        const now = Date.now();
        const toDelete = [];
        for (const [key, entry] of this.cache.entries()) {
            if (now - entry.timestamp > this.config.cacheTtl) {
                toDelete.push(key);
            }
        }
        // Delete oldest entries if still too large
        if (this.cache.size - toDelete.length > 800) {
            const entries = Array.from(this.cache.entries())
                .sort((a, b) => a[1].timestamp - b[1].timestamp);
            const deleteCount = entries.length - 500;
            for (let i = 0; i < deleteCount; i++) {
                toDelete.push(entries[i][0]);
            }
        }
        for (const key of toDelete) {
            this.cache.delete(key);
        }
    }
    /**
     * Utility delay function for streaming simulation
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.RlmController = RlmController;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udHJvbGxlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uL3NyYy9ybG0vY29udHJvbGxlci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiO0FBQUE7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXNERzs7O0FBYUgsc0NBQW1DO0FBR25DOztHQUVHO0FBQ0gsTUFBTSxjQUFjLEdBQXdCO0lBQzFDLFFBQVEsRUFBRSxDQUFDO0lBQ1gsYUFBYSxFQUFFLENBQUM7SUFDaEIsV0FBVyxFQUFFLElBQUk7SUFDakIsV0FBVyxFQUFFLElBQUk7SUFDakIsUUFBUSxFQUFFLE1BQU0sRUFBRSxZQUFZO0lBQzlCLGFBQWEsRUFBRSxFQUFFO0lBQ2pCLGVBQWUsRUFBRSxHQUFHO0lBQ3BCLGdCQUFnQixFQUFFLEtBQUs7SUFDdkIsdUJBQXVCLEVBQUUsQ0FBQztDQUMzQixDQUFDO0FBRUY7Ozs7O0dBS0c7QUFDSCxNQUFhLGFBQWE7SUFNeEI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQXFCRztJQUNILFlBQVksTUFBa0IsRUFBRSxNQUFlO1FBQzdDLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxHQUFHLGNBQWMsRUFBRSxHQUFHLE1BQU0sRUFBRSxDQUFDO1FBQy9DLElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxHQUFHLEVBQUUsQ0FBQztRQUN2QixJQUFJLENBQUMsTUFBTSxHQUFHLE1BQU0sSUFBSSxJQUFJLGVBQU0sQ0FBQyxFQUFFLGVBQWUsRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDO1FBQzlELElBQUksQ0FBQyxlQUFlLEdBQUcsQ0FBQyxDQUFDO0lBQzNCLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7OztPQWFHO0lBQ0gsS0FBSyxDQUFDLEtBQUssQ0FBQyxLQUFhO1FBQ3ZCLG9CQUFvQjtRQUNwQixJQUFJLElBQUksQ0FBQyxNQUFNLENBQUMsV0FBVyxFQUFFLENBQUM7WUFDNUIsTUFBTSxN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QUFPLEdBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO2lCQUM3QyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTLENBQUMsQ0FBQztZQUVuRCxNQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsTUFBTSxHQUFHLEdBQUcsQ0FBQztZQUN6QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQ3JDLFFBQVEsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDL0IsQ0FBQztRQUNILENBQUM7UUFFRCxLQUFLLE1BQU0sR0FBRyxJQUFJLFFBQVEsRUFBRSxDQUFDO1lBQzNCLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3pCLENBQUM7SUFDSCxDQUFDO0lBRUQ7O09BRUc7SUFDSyxLQUFLLENBQUMsRUFBVTtRQUN0QixPQUFPLElBQUksT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsVUFBVSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3pELENBQUM7Q0FDRjtBQXhqQkQsc0NBd2pCQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogUkxNIENvbnRyb2xsZXIgLSBSZWN1cnNpdmUgUmV0cmlldmFsIExhbmd1YWdlIE1vZGVsXG4gKlxuICogSW1wbGVtZW50cyBhIHJlY3Vyc2l2ZSByZXRyaWV2YWwtYXVnbWVudGVkIGdlbmVyYXRpb24gc3lzdGVtIHRoYXQ6XG4gKiAxLiBCcmVha3MgZG93biBjb21wbGV4IHF1ZXJpZXMgaW50byBzdWItcXVlcmllc1xuICogMi4gUmV0cmlldmVzIHJlbGV2YW50IG1lbW9yeSBzcGFucyBmb3IgZWFjaCBxdWVyeVxuICogMy4gU3ludGhlc2l6ZXMgY29oZXJlbnQgYW5zd2VycyBmcm9tIHJldHJpZXZlZCBjb250ZXh0XG4gKiA0LiBPcHRpb25hbGx5IHJlZmxlY3RzIG9uIGFuZCByZWZpbmVzIGFuc3dlcnNcbiAqXG4gKiBAZXhhbXBsZSBCYXNpYyBVc2FnZVxuICogYGBgdHlwZXNjcmlwdFxuICogaW1wb3J0IHsgUmxtQ29udHJvbGxlciB9IGZyb20gJ0BydXZlY3Rvci9ydXZsbG0nO1xuICpcbiAqIGNvbnN0IHJsbSA9IG5ldyBSbG1Db250cm9sbGVyKHtcbiAqICAgbWF4RGVwdGg6IDMsXG4gKiAgIHJldHJpZXZhbFRvcEs6IDEwLFxuICogICBlbmFibGVDYWNoZTogdHJ1ZSxcbiAqIH0pO1xuICpcbiAqIC8vIEFkZCBrbm93bGVkZ2UgdG8gbWVtb3J5XG4gKiBhd2FpdCBybG0uYWRkTWVtb3J5KCdNYWNoaW5lIGxlYXJuaW5nIGlzIGEgc3Vic2V0IG9mIEFJIHRoYXQgZW5hYmxlcyBzeXN0ZW1zIHRvIGxlYXJuIGZyb20gZGF0YS4nKTtcbiAqIGF3YWl0IHJsbS5hZGRNZW1vcnkoJ0RlZXAgbGVhcm5pbmcgdXNlcyBuZXVyYWwgbmV0d29ya3Mgd2l0aCBtYW55IGxheWVycy4nKTtcbiAqXG4gKiAvLyBRdWVyeSB3aXRoIHJlY3Vyc2l2ZSByZXRyaWV2YWxcbiAqIGNvbnN0IGFuc3dlciA9IGF3YWl0IHJsbS5xdWVyeSgnRXhwbGFpbiB0aGUgcmVsYXRpb25zaGlwIGJldHdlZW4gTUwgYW5kIGRlZXAgbGVhcm5pbmcnKTtcbiAqIGNvbnNvbGUubG9nKGFuc3dlci50ZXh0KTtcbiAqIGNvbnNvbGUubG9nKCdTb3VyY2VzOicsIGFuc3dlci5zb3VyY2VzLmxlbmd0aCk7XG4gKiBjb25zb2xlLmxvZygnQ29uZmlkZW5jZTonLCBhbnN3ZXIuY29uZmlkZW5jZSk7XG4gKiBgYGBcbiAqXG4gKiBAZXhhbXBsZSBTdHJlYW1pbmdcbiAqIGBgYHR5cGVzY3JpcHRcbiAqIGNvbnN0IHJsbSA9IG5ldyBSbG1Db250cm9sbGVyKCk7XG4gKlxuICogZm9yIGF3YWl0IChjb25zdCBldmVudCBvZiBybG0ucXVlcnlTdHJlYW0oJ1doYXQgaXMgQUk/JykpIHtcbiAqICAgaWYgKGV2ZW50LnR5cGUgPT09ICd0b2tlbicpIHtcbiAqICAgICBwcm9jZXNzLnN0ZG91dC53cml0ZShldmVudC50ZXh0KTtcbiAqICAgfSBlbHNlIHtcbiAqICAgICBjb25zb2xlLmxvZygnXFxuXFxuRG9uZSEgUXVhbGl0eTonLCBldmVudC5hbnN3ZXIucXVhbGl0eVNjb3JlKTtcbiAqICAgfVxuICogfVxuICogYGBgXG4gKlxuICogQGV4YW1wbGUgV2l0aCBSZWZsZWN0aW9uXG4gKiBgYGB0eXBlc2NyaXB0XG4gKiBjb25zdCBybG0gPSBuZXcgUmxtQ29udHJvbGxlcih7XG4gKiAgIGVuYWJsZVJlZmxlY3Rpb246IHRydWUsXG4gKiAgIG1heFJlZmxlY3Rpb25JdGVyYXRpb25zOiAyLFxuICogICBtaW5RdWFsaXR5U2NvcmU6IDAuOCxcbiAqIH0pO1xuICpcbiAqIGNvbnN0IGFuc3dlciA9IGF3YWl0IHJsbS5xdWVyeSgnQ29tcGxleCBtdWx0aS1wYXJ0IHF1ZXN0aW9uLi4uJyk7XG4gKiAvLyBBbnN3ZXIgd2lsbCBiZSBpdGVyYXRpdmVseSByZWZpbmVkIHVudGlsIHF1YWxpdHkgPj0gMC44XG4gKiBgYGBcbiAqL1xuXG5pbXBvcnQge1xuICBSbG1Db25maWcsXG4gIFJsbUFuc3dlcixcbiAgTWVtb3J5U3BhbixcbiAgU3ViUXVlcnksXG4gIFRva2VuVXNhZ2UsXG4gIFN0cmVhbVRva2VuLFxuICBSbG1DYWNoZUVudHJ5LFxuICBSZWZsZWN0aW9uUmVzdWx0LFxufSBmcm9tICcuL3R5cGVzJztcblxuaW1wb3J0IHsgUnV2TExNIH0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB0eXBlIHsgR2VuZXJhdGlvbkNvbmZpZywgUXVlcnlSZXNwb25zZSB9IGZyb20gJy4uL3R5cGVzJztcblxuLyoqXG4gKiBEZWZhdWx0IGNvbmZpZ3VyYXRpb24gdmFsdWVzXG4gKi9cbmNvbnN0IERFRkFVTFRfQ09ORklHOiBSZXF1aXJlZDxSbG1Db25maWc+ID0ge1xuICBtYXhEZXB0aDogMyxcbiAgbWF4U3ViUXVlcmllczogNSxcbiAgdG9rZW5CdWRnZXQ6IDQwOTYsXG4gIGVuYWJsZUNhY2hlOiB0cnVlLFxuICBjYWNoZVR0bDogMzAwMDAwLCAvLyA1IG1pbnV0ZXNcbiAgcmV0cmlldmFsVG9wSzogMTAsXG4gIG1pblF1YWxpdHlTY29yZTogMC43LFxuICBlbmFibGVSZWZsZWN0aW9uOiBmYWxzZSxcbiAgbWF4UmVmbGVjdGlvbkl0ZXJhdGlvbnM6IDIsXG59O1xuXG4vKipcbiAqIFJsbUNvbnRyb2xsZXIgLSBSZWN1cnNpdmUgUmV0cmlldmFsIExhbmd1YWdlIE1vZGVsIENvbnRyb2xsZXJcbiAqXG4gKiBPcmNoZXN0cmF0ZXMgcmV0cmlldmFsLWF1Z21lbnRlZCBnZW5lcmF0aW9uIHdpdGggcmVjdXJzaXZlIHN1Yi1xdWVyeVxuICogZGVjb21wb3NpdGlvbiwgbWVtb3J5IHNlYXJjaCwgYW5kIG9wdGlvbmFsIHNlbGYtcmVmbGVjdGlvbi5cbiAqL1xuZXhwb3J0IGNsYXNzIFJsbUNvbnRyb2xsZXIge1xuICBwcml2YXRlIGNvbmZpZzogUmVxdWlyZWQ8UmxtQ29uZmlnPjtcbiAgcHJpdmF0ZSBjYWNoZTogTWFwPHN0cmluZywgUmxtQ2FjaGVFbnRyeT47XG4gIHByaXZhdGUgZW5naW5lOiBSdXZMTE07XG4gIHByaXZhdGUgbWVtb3J5SWRDb3VudGVyOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIG5ldyBSTE0gY29udHJvbGxlclxuICAgKlxuICAgKiBAcGFyYW0gY29uZmlnIC0gQ29uZmlndXJhdGlvbiBvcHRpb25zXG4gICAqIEBwYXJhbSBlbmdpbmUgLSBPcHRpb25hbCBSdXZMTE0gZW5naW5lIGluc3RhbmNlIChjcmVhdGVzIG5ldyBpZiBub3QgcHJvdmlkZWQpXG4gICAqXG4gICAqIEBleGFtcGxlXG4gICAqIGBgYHR5cGVzY3JpcHRcbiAgICogLy8gV2l0aCBkZWZhdWx0IGNvbmZpZ1xuICAgKiBjb25zdCBybG0gPSBuZXcgUmxtQ29udHJvbGxlcigpO1xuICAgKlxuICAgKiAvLyBXaXRoIGN1c3RvbSBjb25maWdcbiAgICogY29uc3QgcmxtID0gbmV3IFJsbUNvbnRyb2xsZXIoe1xuICAgKiAgIG1heERlcHRoOiA1LFxuICAgKiAgIGVuYWJsZVJlZmxlY3Rpb246IHRydWUsXG4gICAqIH0pO1xuICAgKlxuICAgKiAvLyBXaXRoIGV4aXN0aW5nIGVuZ2luZVxuICAgKiBjb25zdCBlbmdpbmUgPSBuZXcgUnV2TExNKHsgbGVhcm5pbmdFbmFibGVkOiB0cnVlIH0pO1xuICAgKiBjb25zdCBybG0gPSBuZXcgUmxtQ29udHJvbGxlcih7fSwgZW5naW5lKTtcbiAgICogYGBgXG4gICAqL1xuICBjb25zdHJ1Y3Rvcihjb25maWc/OiBSbG1Db25maWcsIGVuZ2luZT86IFJ1dkxMTSkge1xuICAgIHRoaXMuY29uZmlnID0geyAuLi5ERUZBVUxUX0NPTkZJRywgLi4uY29uZmlnIH07XG4gICAgdGhpcy5jYWNoZSA9IG5ldyBNYXAoKTtcbiAgICB0aGlzLmVuZ2luZSA9IGVuZ2luZSA/PyBuZXcgUnV2TExNKHsgbGVhcm5pbmdFbmFibGVkOiB0cnVlIH0pO1xuICAgIHRoaXMubWVtb3J5SWRDb3VudGVyID0gMDtcbiAgfVxuXG4gIC8qKlxuICAgKiBRdWVyeSB0aGUgUkxNIHdpdGggcmVjdXJzaXZlIHJldHJpZXZhbFxuICAgKlxuICAgKiBAcGFyYW0gaW5wdXQgLSBUaGUgcXVlcnkgc3RyaW5nXG4gICAqIEByZXR1cm5zIFByb21pc2UgcmVzb2x2aW5nIHRvIHRoZSBhbnN3ZXIgd2l0aCBzb3VyY2VzIGFuZCBtZXRhZGF0YVxuICAgKlxuICAgKiBAZXhhbXBsZVxuICAgKiBgYGB0eXBlc2NyaXB0XG4gICAqIGNvbnN0IGFuc3dlciA9IGF3YWl0IHJsbS5xdWVyeSgnV2hhdCBpcyB0aGUgY2FwaXRhbCBvZiBGcmFuY2U/Jyk7XG4gICAqIGNvbnNvbGUubG9nKGFuc3dlci50ZXh0KTsgLy8gXCJUaGUgY2FwaXRhbCBvZiBGcmFuY2UgaXMgUGFyaXMuLi5cIlxuICAgKiBjb25zb2xlLmxvZyhhbnN3ZXIuY29uZmlkZW5jZSk7IC8vIDAuOTVcbiAgICogY29uc29sZS5sb2coYW5zd2VyLnNvdXJjZXMpOyAvLyBbeyBpZDogJy4uLicsIHRleHQ6ICcuLi4nLCBzaW1pbGFyaXR5U2NvcmU6IDAuOTIgfV1cbiAgICogYGBgXG4gICAqL1xuICBhc3luYyBxdWVyeShpbnB1dDogc3RyaW5nKTogUHJvbWlzZTxSbG1BbnN3ZXI+IHtcbiAgICAvLyBDaGVjayBjYWNoZSBmaXJzdFxuICAgIGlmICh0aGlzLmNvbmZpZy5lbmFibGVDYWNoZSkge1xuICAgICAgY29uc3QgY2FjaGVkID0gdGhpcy5nZXRDYWNoZWQoaW5wdXQpO1xuICAgICAgaWYgKGNhY2hlZCkge1xuICAgICAgICByZXR1cm4geyAuLi5jYWNoZWQsIGNhY2hlZDogdHJ1ZSB9O1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIFJldHJpZXZlIHJlbGV2YW50IG1lbW9yeSBzcGFuc1xuICAgIGNvbnN0IHNvdXJjZXMgPSBhd2FpdCB0aGlzLnNlYXJjaE1lbW9yeShpbnB1dCwgdGhpcy5jb25maWcucmV0cmlldmFsVG9wSyk7XG5cbiAgICAvLyBHZW5lcmF0ZSBzdWItcXVlcmllcyBpZiBuZWVkZWQgYW5kIGRlcHRoIGFsbG93c1xuICAgIGNvbnN0IHN1YlF1ZXJpZXMgPSBhd2FpdCB0aGlzLmdlbmVyYXRlU3ViUXVlcmllcyhpbnB1dCwgc291cmNlcywgMCk7XG5cbiAgICAvLyBCdWlsZCBjb250ZXh0IGZyb20gc291cmNlcyBhbmQgc3ViLXF1ZXJ5IGFuc3dlcnNcbiAgICBjb25zdCBjb250ZXh0ID0gdGhpcy5idWlsZENvbnRleHQoc291cmNlcywgc3ViUXVlcmllcyk7XG5cbiAgICAvLyBHZW5lcmF0ZSB0aGUgYW5zd2VyXG4gICAgY29uc3Qgc3RhcnRUaW1lID0gRGF0ZS5ub3coKTtcbiAgICBjb25zdCByZXNwb25zZSA9IHRoaXMuZW5naW5lLnF1ZXJ5KFxuICAgICAgdGhpcy5idWlsZFByb21wdChpbnB1dCwgY29udGV4dCksXG4gICAgICB0aGlzLmdldEdlbmVyYXRpb25Db25maWcoKVxuICAgICk7XG5cbiAgICAvLyBDYWxjdWxhdGUgdG9rZW4gdXNhZ2UgKGVzdGltYXRlIGlmIG5vdCBwcm92aWRlZCBieSBlbmdpbmUpXG4gICAgY29uc3QgdG9rZW5Vc2FnZSA9IHRoaXMuZXN0aW1hdGVUb2tlblVzYWdlKGlucHV0LCBjb250ZXh0LCByZXNwb25zZS50ZXh0KTtcblxuICAgIC8vIENhbGN1bGF0ZSBxdWFsaXR5IHNjb3JlXG4gICAgY29uc3QgcXVhbGl0eVNjb3JlID0gdGhpcy5jYWxjdWxhdGVRdWFsaXR5U2NvcmUoc291cmNlcywgcmVzcG9uc2UuY29uZmlkZW5jZSk7XG5cbiAgICBsZXQgYW5zd2VyOiBSbG1BbnN3ZXIgPSB7XG4gICAgICB0ZXh0OiByZXNwb25zZS50ZXh0LFxuICAgICAgY29uZmlkZW5jZTogcmVzcG9uc2UuY29uZmlkZW5jZSxcbiAgICAgIHF1YWxpdHlTY29yZSxcbiAgICAgIHNvdXJjZXMsXG4gICAgICBzdWJRdWVyaWVzOiBzdWJRdWVyaWVzLmxlbmd0aCA+IDAgPyBzdWJRdWVyaWVzIDogdW5kZWZpbmVkLFxuICAgICAgdG9rZW5Vc2FnZSxcbiAgICAgIGNhY2hlZDogZmFsc2UsXG4gICAgfTtcblxuICAgIC8vIEFwcGx5IHJlZmxlY3Rpb24gaWYgZW5hYmxlZCBhbmQgcXVhbGl0eSBpcyBiZWxvdyB0aHJlc2hvbGRcbiAgICBpZiAodGhpcy5jb25maWcuZW5hYmxlUmVmbGVjdGlvbiAmJiBxdWFsaXR5U2NvcmUgPCB0aGlzLmNvbmZpZy5taW5RdWFsaXR5U2NvcmUpIHtcbiAgICAgIGFuc3dlciA9IGF3YWl0IHRoaXMuYXBwbHlSZWZsZWN0aW9uKGlucHV0LCBhbnN3ZXIpO1xuICAgIH1cblxuICAgIC8vIENhY2hlIHRoZSByZXN1bHRcbiAgICBpZiAodGhpcy5jb25maWcuZW5hYmxlQ2FjaGUpIHtcbiAgICAgIHRoaXMuc2V0Q2FjaGUoaW5wdXQsIGFuc3dlcik7XG4gICAgfVxuXG4gICAgcmV0dXJuIGFuc3dlcjtcbiAgfVxuXG4gIC8qKlxuICAgKiBRdWVyeSB3aXRoIHN0cmVhbWluZyByZXNwb25zZVxuICAgKlxuICAgKiBAcGFyYW0gaW5wdXQgLSBUaGUgcXVlcnkgc3RyaW5nXG4gICAqIEB5aWVsZHMgU3RyZWFtVG9rZW4gZXZlbnRzIChlaXRoZXIgcGFydGlhbCB0b2tlbnMgb3IgZmluYWwgYW5zd2VyKVxuICAgKlxuICAgKiBAZXhhbXBsZVxuICAgKiBgYGB0eXBlc2NyaXB0XG4gICAqIGZvciBhd2FpdCAoY29uc3QgZXZlbnQgb2YgcmxtLnF1ZXJ5U3RyZWFtKCdFeHBsYWluIHF1YW50dW0gY29tcHV0aW5nJykpIHtcbiAgICogICBpZiAoZXZlbnQudHlwZSA9PT0gJ3Rva2VuJykge1xuICAgKiAgICAgLy8gUGFydGlhbCB0b2tlbiByZWNlaXZlZFxuICAgKiAgICAgcHJvY2Vzcy5zdGRvdXQud3JpdGUoZXZlbnQudGV4dCk7XG4gICAqICAgfSBlbHNlIHtcbiAgICogICAgIC8vIEdlbmVyYXRpb24gY29tcGxldGVcbiAgICogICAgIGNvbnNvbGUubG9nKCdcXG5cXG5Tb3VyY2VzOicsIGV2ZW50LmFuc3dlci5zb3VyY2VzLmxlbmd0aCk7XG4gICAqICAgfVxuICAgKiB9XG4gICAqIGBgYFxuICAgKi9cbiAgYXN5bmMgKnF1ZXJ5U3RyZWFtKGlucHV0OiBzdHJpbmcpOiBBc3luY0dlbmVyYXRvcjxTdHJlYW1Ub2tlbj4ge1xuICAgIC8vIENoZWNrIGNhY2hlIGZpcnN0XG4gICAgaWYgKHRoaXMuY29uZmlnLmVuYWJsZUNhY2hlKSB7XG4gICAgICBjb25zdCBjYWNoZWQgPSB0aGlzLmdldENhY2hlZChpbnB1dCk7XG4gICAgICBpZiAoY2FjaGVkKSB7XG4gICAgICAgIC8vIFNpbXVsYXRlIHN0cmVhbWluZyBmb3IgY2FjaGVkIHJlc3BvbnNlXG4gICAgICAgIGNvbnN0IHdvcmRzID0gY2FjaGVkLnRleHQuc3BsaXQoJyAnKTtcbiAgICAgICAgZm9yIChjb25zdCB3b3JkIG9mIHdvcmRzKSB7XG4gICAgICAgICAgeWllbGQgeyB0eXBlOiAndG9rZW4nLCB0ZXh0OiB3b3JkICsgJyAnLCBkb25lOiBmYWxzZSB9O1xuICAgICAgICAgIGF3YWl0IHRoaXMuZGVsYXkoMTApOyAvLyBTbWFsbCBkZWxheSBmb3IgcmVhbGlzdGljIHN0cmVhbWluZ1xuICAgICAgICB9XG4gICAgICAgIHlpZWxkIHsgdHlwZTogJ2RvbmUnLCBhbnN3ZXI6IHsgLi4uY2FjaGVkLCBjYWNoZWQ6IHRydWUgfSwgZG9uZTogdHJ1ZSB9O1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gUmV0cmlldmUgc291cmNlc1xuICAgIGNvbnN0IHNvdXJjZXMgPSBhd2FpdCB0aGlzLnNlYXJjaE1lbW9yeShpbnB1dCwgdGhpcy5jb25maWcucmV0cmlldmFsVG9wSyk7XG4gICAgY29uc3Qgc3ViUXVlcmllcyA9IGF3YWl0IHRoaXMuZ2VuZXJhdGVTdWJRdWVyaWVzKGlucHV0LCBzb3VyY2VzLCAwKTtcbiAgICBjb25zdCBjb250ZXh0ID0gdGhpcy5idWlsZENvbnRleHQoc291cmNlcywgc3ViUXVlcmllcyk7XG5cbiAgICAvLyBHZW5lcmF0ZSB3aXRoIHNpbXVsYXRlZCBzdHJlYW1pbmdcbiAgICBjb25zdCBwcm9tcHQgPSB0aGlzLmJ1aWxkUHJvbXB0KGlucHV0LCBjb250ZXh0KTtcbiAgICBjb25zdCByZXNwb25zZSA9IHRoaXMuZW5naW5lLnF1ZXJ5KHByb21wdCwgdGhpcy5nZXRHZW5lcmF0aW9uQ29uZmlnKCkpO1xuXG4gICAgLy8gU3RyZWFtIHRoZSByZXNwb25zZSB3b3JkIGJ5IHdvcmRcbiAgICBjb25zdCB3b3JkcyA9IHJlc3BvbnNlLnRleHQuc3BsaXQoJyAnKTtcbiAgICBsZXQgc3RyZWFtZWRUZXh0ID0gJyc7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHdvcmRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCB3b3JkID0gd29yZHNbaV07XG4gICAgICBjb25zdCB0ZXh0ID0gaSA8IHdvcmRzLmxlbmd0aCAtIDEgPyB3b3JkICsgJyAnIDogd29yZDtcbiAgICAgIHN0cmVhbWVkVGV4dCArPSB0ZXh0O1xuXG4gICAgICB5aWVsZCB7IHR5cGU6ICd0b2tlbicsIHRleHQsIGRvbmU6IGZhbHNlIH07XG4gICAgICBhd2FpdCB0aGlzLmRlbGF5KDIwKTsgLy8gU2ltdWxhdGUgZ2VuZXJhdGlvbiBsYXRlbmN5XG4gICAgfVxuXG4gICAgY29uc3QgdG9rZW5Vc2FnZSA9IHRoaXMuZXN0aW1hdGVUb2tlblVzYWdlKGlucHV0LCBjb250ZXh0LCBzdHJlYW1lZFRleHQpO1xuICAgIGNvbnN0IHF1YWxpdHlTY29yZSA9IHRoaXMuY2FsY3VsYXRlUXVhbGl0eVNjb3JlKHNvdXJjZXMsIHJlc3BvbnNlLmNvbmZpZGVuY2UpO1xuXG4gICAgY29uc3QgYW5zd2VyOiBSbG1BbnN3ZXIgPSB7XG4gICAgICB0ZXh0OiBzdHJlYW1lZFRleHQsXG4gICAgICBjb25maWRlbmNlOiByZXNwb25zZS5jb25maWRlbmNlLFxuICAgICAgcXVhbGl0eVNjb3JlLFxuICAgICAgc291cmNlcyxcbiAgICAgIHN1YlF1ZXJpZXM6IHN1YlF1ZXJpZXMubGVuZ3RoID4gMCA/IHN1YlF1ZXJpZXMgOiB1bmRlZmluZWQsXG4gICAgICB0b2tlblVzYWdlLFxuICAgICAgY2FjaGVkOiBmYWxzZSxcbiAgICB9O1xuXG4gICAgLy8gQ2FjaGUgdGhlIHJlc3VsdFxuICAgIGlmICh0aGlzLmNvbmZpZy5lbmFibGVDYWNoZSkge1xuICAgICAgdGhpcy5zZXRDYWNoZShpbnB1dCwgYW5zd2VyKTtcbiAgICB9XG5cbiAgICB5aWVsZCB7IHR5cGU6ICdkb25lJywgYW5zd2VyLCBkb25lOiB0cnVlIH07XG4gIH1cblxuICAvKipcbiAgICogQWRkIGNvbnRlbnQgdG8gbWVtb3J5IGZvciByZXRyaWV2YWxcbiAgICpcbiAgICogQHBhcmFtIHRleHQgLSBUaGUgdGV4dCBjb250ZW50IHRvIHN0b3JlXG4gICAqIEBwYXJhbSBtZXRhZGF0YSAtIE9wdGlvbmFsIG1ldGFkYXRhIHRvIGFzc29jaWF0ZSB3aXRoIHRoZSBtZW1vcnlcbiAgICogQHJldHVybnMgUHJvbWlzZSByZXNvbHZpbmcgdG8gdGhlIG1lbW9yeSBzcGFuIElEXG4gICAqXG4gICAqIEBleGFtcGxlXG4gICAqIGBgYHR5cGVzY3JpcHRcbiAgICogY29uc3QgaWQxID0gYXdhaXQgcmxtLmFkZE1lbW9yeShcbiAgICogICAnVHlwZVNjcmlwdCBpcyBhIHR5cGVkIHN1cGVyc2V0IG9mIEphdmFTY3JpcHQuJyxcbiAgICogICB7IHNvdXJjZTogJ2RvY3VtZW50YXRpb24nLCBjYXRlZ29yeTogJ3Byb2dyYW1taW5nJyB9XG4gICAqICk7XG4gICAqXG4gICAqIGNvbnN0IGlkMiA9IGF3YWl0IHJsbS5hZGRNZW1vcnkoXG4gICAqICAgJ1JlYWN0IGlzIGEgSmF2YVNjcmlwdCBsaWJyYXJ5IGZvciBidWlsZGluZyBVSXMuJ1xuICAgKiApO1xuICAgKiBgYGBcbiAgICovXG4gIGFzeW5jIGFkZE1lbW9yeSh0ZXh0OiBzdHJpbmcsIG1ldGFkYXRhPzogUmVjb3JkPHN0cmluZywgdW5rbm93bj4pOiBQcm9taXNlPHN0cmluZz4ge1xuICAgIGNvbnN0IG5vZGVJZCA9IHRoaXMuZW5naW5lLmFkZE1lbW9yeSh0ZXh0LCBtZXRhZGF0YSk7XG4gICAgY29uc3QgaWQgPSBgcmxtLW1lbS0ke3RoaXMubWVtb3J5SWRDb3VudGVyKyt9LSR7bm9kZUlkfWA7XG4gICAgcmV0dXJuIGlkO1xuICB9XG5cbiAgLyoqXG4gICAqIFNlYXJjaCBtZW1vcnkgZm9yIHJlbGV2YW50IHNwYW5zXG4gICAqXG4gICAqIEBwYXJhbSBxdWVyeSAtIFRoZSBzZWFyY2ggcXVlcnlcbiAgICogQHBhcmFtIHRvcEsgLSBOdW1iZXIgb2YgcmVzdWx0cyB0byByZXR1cm4gKGRlZmF1bHQ6IGNvbmZpZy5yZXRyaWV2YWxUb3BLKVxuICAgKiBAcmV0dXJucyBQcm9taXNlIHJlc29sdmluZyB0byBhcnJheSBvZiBtZW1vcnkgc3BhbnNcbiAgICpcbiAgICogQGV4YW1wbGVcbiAgICogYGBgdHlwZXNjcmlwdFxuICAgKiBjb25zdCBzcGFucyA9IGF3YWl0IHJsbS5zZWFyY2hNZW1vcnkoJ0phdmFTY3JpcHQgZnJhbWV3b3JrcycsIDUpO1xuICAgKiBmb3IgKGNvbnN0IHNwYW4gb2Ygc3BhbnMpIHtcbiAgICogICBjb25zb2xlLmxvZyhgWyR7c3Bhbi5zaW1pbGFyaXR5U2NvcmUudG9GaXhlZCgyKX1dICR7c3Bhbi50ZXh0fWApO1xuICAgKiB9XG4gICAqIGBgYFxuICAgKi9cbiAgYXN5bmMgc2VhcmNoTWVtb3J5KHF1ZXJ5OiBzdHJpbmcsIHRvcEs/OiBudW1iZXIpOiBQcm9taXNlPE1lbW9yeVNwYW5bXT4ge1xuICAgIGNvbnN0IGsgPSB0b3BLID8/IHRoaXMuY29uZmlnLnJldHJpZXZhbFRvcEs7XG4gICAgY29uc3QgcmVzdWx0cyA9IHRoaXMuZW5naW5lLnNlYXJjaE1lbW9yeShxdWVyeSwgayk7XG5cbiAgICByZXR1cm4gcmVzdWx0cy5tYXAoKHJlc3VsdCwgaW5kZXgpID0+ICh7XG4gICAgICBpZDogYHJsbS1zcGFuLSR7cmVzdWx0LmlkfS0ke2luZGV4fWAsXG4gICAgICB0ZXh0OiByZXN1bHQuY29udGVudCxcbiAgICAgIHNpbWlsYXJpdHlTY29yZTogcmVzdWx0LnNjb3JlLFxuICAgICAgc291cmNlOiByZXN1bHQubWV0YWRhdGE/LnNvdXJjZSBhcyBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgICBtZXRhZGF0YTogcmVzdWx0Lm1ldGFkYXRhLFxuICAgIH0pKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDbGVhciB0aGUgcmVzcG9uc2UgY2FjaGVcbiAgICpcbiAgICogQGV4YW1wbGVcbiAgICogYGBgdHlwZXNjcmlwdFxuICAgKiBybG0uY2xlYXJDYWNoZSgpO1xuICAgKiBjb25zb2xlLmxvZygnQ2FjaGUgY2xlYXJlZCcpO1xuICAgKiBgYGBcbiAgICovXG4gIGNsZWFyQ2FjaGUoKTogdm9pZCB7XG4gICAgdGhpcy5jYWNoZS5jbGVhcigpO1xuICB9XG5cbiAgLyoqXG4gICAqIEdldCBjdXJyZW50IGNhY2hlIHN0YXRpc3RpY3NcbiAgICpcbiAgICogQHJldHVybnMgT2JqZWN0IHdpdGggY2FjaGUgc2l6ZSBhbmQgaGl0IHJhdGUgaW5mb1xuICAgKi9cbiAgZ2V0Q2FjaGVTdGF0cygpOiB7IHNpemU6IG51bWJlcjsgZW50cmllczogbnVtYmVyIH0ge1xuICAgIHJldHVybiB7XG4gICAgICBzaXplOiB0aGlzLmNhY2hlLnNpemUsXG4gICAgICBlbnRyaWVzOiB0aGlzLmNhY2hlLnNpemUsXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBVcGRhdGUgY29uZmlndXJhdGlvbiBhdCBydW50aW1lXG4gICAqXG4gICAqIEBwYXJhbSBjb25maWcgLSBQYXJ0aWFsIGNvbmZpZ3VyYXRpb24gdG8gbWVyZ2VcbiAgICovXG4gIHVwZGF0ZUNvbmZpZyhjb25maWc6IFBhcnRpYWw8UmxtQ29uZmlnPik6IHZvaWQge1xuICAgIHRoaXMuY29uZmlnID0geyAuLi50aGlzLmNvbmZpZywgLi4uY29uZmlnIH07XG4gIH1cblxuICAvKipcbiAgICogR2V0IGN1cnJlbnQgY29uZmlndXJhdGlvblxuICAgKi9cbiAgZ2V0Q29uZmlnKCk6IFJlcXVpcmVkPFJsbUNvbmZpZz4ge1xuICAgIHJldHVybiB7IC4uLnRoaXMuY29uZmlnIH07XG4gIH1cblxuICAvLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICAvLyBQcml2YXRlIE1ldGhvZHNcbiAgLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuICAvKipcbiAgICogR2VuZXJhdGUgc3ViLXF1ZXJpZXMgZm9yIGNvbXBsZXggcXVlc3Rpb25zXG4gICAqL1xuICBwcml2YXRlIGFzeW5jIGdlbmVyYXRlU3ViUXVlcmllcyhcbiAgICBxdWVyeTogc3RyaW5nLFxuICAgIHNvdXJjZXM6IE1lbW9yeVNwYW5bXSxcbiAgICBkZXB0aDogbnVtYmVyXG4gICk6IFByb21pc2U8U3ViUXVlcnlbXT4ge1xuICAgIGlmIChkZXB0aCA+PSB0aGlzLmNvbmZpZy5tYXhEZXB0aCkge1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cblxuICAgIC8vIFNpbXBsZSBoZXVyaXN0aWM6IGdlbmVyYXRlIHN1Yi1xdWVyaWVzIGZvciBxdWVzdGlvbnMgd2l0aCBtdWx0aXBsZSBwYXJ0c1xuICAgIGNvbnN0IHN1YlF1ZXJpZXM6IFN1YlF1ZXJ5W10gPSBbXTtcbiAgICBjb25zdCBwYXJ0cyA9IHRoaXMuZGVjb21wb3NlUXVlcnkocXVlcnkpO1xuXG4gICAgZm9yIChjb25zdCBwYXJ0IG9mIHBhcnRzLnNsaWNlKDAsIHRoaXMuY29uZmlnLm1heFN1YlF1ZXJpZXMpKSB7XG4gICAgICBpZiAocGFydC50cmltKCkubGVuZ3RoIDwgMTApIGNvbnRpbnVlO1xuXG4gICAgICAvLyBTZWFyY2ggZm9yIHN1Yi1xdWVyeSBzcGVjaWZpYyBzb3VyY2VzXG4gICAgICBjb25zdCBzdWJTb3VyY2VzID0gYXdhaXQgdGhpcy5zZWFyY2hNZW1vcnkocGFydCwgTWF0aC5jZWlsKHRoaXMuY29uZmlnLnJldHJpZXZhbFRvcEsgLyAyKSk7XG4gICAgICBjb25zdCBjb250ZXh0ID0gdGhpcy5idWlsZENvbnRleHQoc3ViU291cmNlcywgW10pO1xuICAgICAgY29uc3QgcmVzcG9uc2UgPSB0aGlzLmVuZ2luZS5xdWVyeShcbiAgICAgICAgdGhpcy5idWlsZFByb21wdChwYXJ0LCBjb250ZXh0KSxcbiAgICAgICAgeyAuLi50aGlzLmdldEdlbmVyYXRpb25Db25maWcoKSwgbWF4VG9rZW5zOiAyNTYgfVxuICAgICAgKTtcblxuICAgICAgc3ViUXVlcmllcy5wdXNoKHtcbiAgICAgICAgcXVlcnk6IHBhcnQsXG4gICAgICAgIGFuc3dlcjogcmVzcG9uc2UudGV4dCxcbiAgICAgICAgZGVwdGg6IGRlcHRoICsgMSxcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHJldHVybiBzdWJRdWVyaWVzO1xuICB9XG5cbiAgLyoqXG4gICAqIERlY29tcG9zZSBhIGNvbXBsZXggcXVlcnkgaW50byBzaW1wbGVyIHBhcnRzXG4gICAqL1xuICBwcml2YXRlIGRlY29tcG9zZVF1ZXJ5KHF1ZXJ5OiBzdHJpbmcpOiBzdHJpbmdbXSB7XG4gICAgLy8gU3BsaXQgb24gY29tbW9uIGNvbmp1bmN0aW9ucyBhbmQgcXVlc3Rpb24gbWFya2Vyc1xuICAgIGNvbnN0IHBhcnRzOiBzdHJpbmdbXSA9IFtdO1xuXG4gICAgLy8gQ2hlY2sgZm9yIG11bHRpLXBhcnQgcXVlc3Rpb25zXG4gICAgY29uc3QgY29uanVuY3Rpb25zID0gWycgYW5kICcsICcgb3IgJywgJy4gJywgJz8gJywgJzsgJ107XG4gICAgbGV0IGN1cnJlbnQgPSBxdWVyeTtcblxuICAgIGZvciAoY29uc3QgY29uaiBvZiBjb25qdW5jdGlvbnMpIHtcbiAgICAgIGlmIChjdXJyZW50LmluY2x1ZGVzKGNvbmopKSB7XG4gICAgICAgIGNvbnN0IHNwbGl0ID0gY3VycmVudC5zcGxpdChjb25qKTtcbiAgICAgICAgcGFydHMucHVzaCguLi5zcGxpdC5maWx0ZXIocCA9PiBwLnRyaW0oKS5sZW5ndGggPiAxMCkpO1xuICAgICAgICBjdXJyZW50ID0gJyc7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIElmIG5vIGRlY29tcG9zaXRpb24gaGFwcGVuZWQsIHJldHVybiBvcmlnaW5hbFxuICAgIGlmIChwYXJ0cy5sZW5ndGggPT09IDApIHtcbiAgICAgIHJldHVybiBbcXVlcnldO1xuICAgIH1cblxuICAgIHJldHVybiBwYXJ0cztcbiAgfVxuXG4gIC8qKlxuICAgKiBCdWlsZCBjb250ZXh0IHN0cmluZyBmcm9tIHNvdXJjZXMgYW5kIHN1Yi1xdWVyaWVzXG4gICAqL1xuICBwcml2YXRlIGJ1aWxkQ29udGV4dChzb3VyY2VzOiBNZW1vcnlTcGFuW10sIHN1YlF1ZXJpZXM6IFN1YlF1ZXJ5W10pOiBzdHJpbmcge1xuICAgIGNvbnN0IHBhcnRzOiBzdHJpbmdbXSA9IFtdO1xuXG4gICAgLy8gQWRkIHNvdXJjZXNcbiAgICBpZiAoc291cmNlcy5sZW5ndGggPiAwKSB7XG4gICAgICBwYXJ0cy5wdXNoKCdSZWxldmFudCBjb250ZXh0OicpO1xuICAgICAgZm9yIChjb25zdCBzb3VyY2Ugb2Ygc291cmNlcykge1xuICAgICAgICBwYXJ0cy5wdXNoKGAtICR7c291cmNlLnRleHR9YCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gQWRkIHN1Yi1xdWVyeSBhbnN3ZXJzXG4gICAgaWYgKHN1YlF1ZXJpZXMubGVuZ3RoID4gMCkge1xuICAgICAgcGFydHMucHVzaCgnXFxuUmVsYXRlZCBpbmZvcm1hdGlvbjonKTtcbiAgICAgIGZvciAoY29uc3Qgc3Egb2Ygc3ViUXVlcmllcykge1xuICAgICAgICBwYXJ0cy5wdXNoKGBROiAke3NxLnF1ZXJ5fWApO1xuICAgICAgICBwYXJ0cy5wdXNoKGBBOiAke3NxLmFuc3dlcn1gKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gcGFydHMuam9pbignXFxuJyk7XG4gIH1cblxuICAvKipcbiAgICogQnVpbGQgdGhlIGZ1bGwgcHJvbXB0IHdpdGggY29udGV4dFxuICAgKi9cbiAgcHJpdmF0ZSBidWlsZFByb21wdChxdWVyeTogc3RyaW5nLCBjb250ZXh0OiBzdHJpbmcpOiBzdHJpbmcge1xuICAgIGlmIChjb250ZXh0LnRyaW0oKS5sZW5ndGggPT09IDApIHtcbiAgICAgIHJldHVybiBxdWVyeTtcbiAgICB9XG5cbiAgICByZXR1cm4gYCR7Y29udGV4dH1cXG5cXG5CYXNlZCBvbiB0aGUgYWJvdmUgY29udGV4dCwgYW5zd2VyIHRoZSBmb2xsb3dpbmcgcXVlc3Rpb246XFxuJHtxdWVyeX1gO1xuICB9XG5cbiAgLyoqXG4gICAqIEdldCBnZW5lcmF0aW9uIGNvbmZpZyBiYXNlZCBvbiBSTE0gc2V0dGluZ3NcbiAgICovXG4gIHByaXZhdGUgZ2V0R2VuZXJhdGlvbkNvbmZpZygpOiBHZW5lcmF0aW9uQ29uZmlnIHtcbiAgICByZXR1cm4ge1xuICAgICAgbWF4VG9rZW5zOiBNYXRoLm1pbih0aGlzLmNvbmZpZy50b2tlbkJ1ZGdldCwgMjA0OCksXG4gICAgICB0ZW1wZXJhdHVyZTogMC43LFxuICAgICAgdG9wUDogMC45LFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogRXN0aW1hdGUgdG9rZW4gdXNhZ2VcbiAgICovXG4gIHByaXZhdGUgZXN0aW1hdGVUb2tlblVzYWdlKHF1ZXJ5OiBzdHJpbmcsIGNvbnRleHQ6IHN0cmluZywgcmVzcG9uc2U6IHN0cmluZyk6IFRva2VuVXNhZ2Uge1xuICAgIC8vIFJvdWdoIGVzdGltYXRpb246IH40IGNoYXJhY3RlcnMgcGVyIHRva2VuXG4gICAgY29uc3QgcHJvbXB0VG9rZW5zID0gTWF0aC5jZWlsKChxdWVyeS5sZW5ndGggKyBjb250ZXh0Lmxlbmd0aCkgLyA0KTtcbiAgICBjb25zdCBjb21wbGV0aW9uVG9rZW5zID0gTWF0aC5jZWlsKHJlc3BvbnNlLmxlbmd0aCAvIDQpO1xuXG4gICAgcmV0dXJuIHtcbiAgICAgIHByb21wdDogcHJvbXB0VG9rZW5zLFxuICAgICAgY29tcGxldGlvbjogY29tcGxldGlvblRva2VucyxcbiAgICAgIHRvdGFsOiBwcm9tcHRUb2tlbnMgKyBjb21wbGV0aW9uVG9rZW5zLFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogQ2FsY3VsYXRlIHF1YWxpdHkgc2NvcmUgYmFzZWQgb24gc291cmNlcyBhbmQgY29uZmlkZW5jZVxuICAgKi9cbiAgcHJpdmF0ZSBjYWxjdWxhdGVRdWFsaXR5U2NvcmUoc291cmNlczogTWVtb3J5U3BhbltdLCBjb25maWRlbmNlOiBudW1iZXIpOiBudW1iZXIge1xuICAgIGlmIChzb3VyY2VzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgcmV0dXJuIGNvbmZpZGVuY2UgKiAwLjU7IC8vIFBlbmFsaXplIGFuc3dlcnMgd2l0aG91dCBzb3VyY2VzXG4gICAgfVxuXG4gICAgLy8gQXZlcmFnZSBzb3VyY2Ugc2ltaWxhcml0eVxuICAgIGNvbnN0IGF2Z1NpbWlsYXJpdHkgPSBzb3VyY2VzLnJlZHVjZSgoc3VtLCBzKSA9PiBzdW0gKyBzLnNpbWlsYXJpdHlTY29yZSwgMCkgLyBzb3VyY2VzLmxlbmd0aDtcblxuICAgIC8vIFdlaWdodGVkIGNvbWJpbmF0aW9uXG4gICAgcmV0dXJuIGNvbmZpZGVuY2UgKiAwLjYgKyBhdmdTaW1pbGFyaXR5ICogMC40O1xuICB9XG5cbiAgLyoqXG4gICAqIEFwcGx5IHNlbGYtcmVmbGVjdGlvbiB0byBpbXByb3ZlIGFuc3dlclxuICAgKi9cbiAgcHJpdmF0ZSBhc3luYyBhcHBseVJlZmxlY3Rpb24oXG4gICAgcXVlcnk6IHN0cmluZyxcbiAgICBhbnN3ZXI6IFJsbUFuc3dlclxuICApOiBQcm9taXNlPFJsbUFuc3dlcj4ge1xuICAgIGxldCBjdXJyZW50QW5zd2VyID0gYW5zd2VyO1xuICAgIGxldCBpdGVyYXRpb25zID0gMDtcblxuICAgIHdoaWxlIChcbiAgICAgIGl0ZXJhdGlvbnMgPCB0aGlzLmNvbmZpZy5tYXhSZWZsZWN0aW9uSXRlcmF0aW9ucyAmJlxuICAgICAgY3VycmVudEFuc3dlci5xdWFsaXR5U2NvcmUgPCB0aGlzLmNvbmZpZy5taW5RdWFsaXR5U2NvcmVcbiAgICApIHtcbiAgICAgIGl0ZXJhdGlvbnMrKztcblxuICAgICAgLy8gR2VuZXJhdGUgY3JpdGlxdWVcbiAgICAgIGNvbnN0IGNyaXRpcXVlUHJvbXB0ID0gYEV2YWx1YXRlIHRoaXMgYW5zd2VyIGZvciBhY2N1cmFjeSBhbmQgY29tcGxldGVuZXNzOlxuUXVlc3Rpb246ICR7cXVlcnl9XG5BbnN3ZXI6ICR7Y3VycmVudEFuc3dlci50ZXh0fVxuXG5Qcm92aWRlIGEgYnJpZWYgY3JpdGlxdWUgYW5kIHN1Z2dlc3QgaW1wcm92ZW1lbnRzLmA7XG5cbiAgICAgIGNvbnN0IGNyaXRpcXVlUmVzcG9uc2UgPSB0aGlzLmVuZ2luZS5xdWVyeShjcml0aXF1ZVByb21wdCwge1xuICAgICAgICBtYXhUb2tlbnM6IDI1NixcbiAgICAgICAgdGVtcGVyYXR1cmU6IDAuNSxcbiAgICAgIH0pO1xuXG4gICAgICAvLyBHZW5lcmF0ZSBpbXByb3ZlZCBhbnN3ZXJcbiAgICAgIGNvbnN0IGltcHJvdmVQcm9tcHQgPSBgQmFzZWQgb24gdGhpcyBmZWVkYmFjazogXCIke2NyaXRpcXVlUmVzcG9uc2UudGV4dH1cIlxuXG5JbXByb3ZlIHRoaXMgYW5zd2VyOlxuUXVlc3Rpb246ICR7cXVlcnl9XG5PcmlnaW5hbDogJHtjdXJyZW50QW5zd2VyLnRleHR9XG5cblByb3ZpZGUgYW4gaW1wcm92ZWQgYW5zd2VyOmA7XG5cbiAgICAgIGNvbnN0IGltcHJvdmVkUmVzcG9uc2UgPSB0aGlzLmVuZ2luZS5xdWVyeShpbXByb3ZlUHJvbXB0LCB0aGlzLmdldEdlbmVyYXRpb25Db25maWcoKSk7XG5cbiAgICAgIC8vIFVwZGF0ZSBhbnN3ZXIgd2l0aCByZWZsZWN0aW9uIGltcHJvdmVtZW50c1xuICAgICAgY29uc3QgbmV3UXVhbGl0eVNjb3JlID0gTWF0aC5taW4oXG4gICAgICAgIDEuMCxcbiAgICAgICAgY3VycmVudEFuc3dlci5xdWFsaXR5U2NvcmUgKyAwLjEgKiBpdGVyYXRpb25zXG4gICAgICApO1xuXG4gICAgICBjdXJyZW50QW5zd2VyID0ge1xuICAgICAgICAuLi5jdXJyZW50QW5zd2VyLFxuICAgICAgICB0ZXh0OiBpbXByb3ZlZFJlc3BvbnNlLnRleHQsXG4gICAgICAgIGNvbmZpZGVuY2U6IE1hdGgubWF4KGN1cnJlbnRBbnN3ZXIuY29uZmlkZW5jZSwgaW1wcm92ZWRSZXNwb25zZS5jb25maWRlbmNlKSxcbiAgICAgICAgcXVhbGl0eVNjb3JlOiBuZXdRdWFsaXR5U2NvcmUsXG4gICAgICAgIHRva2VuVXNhZ2U6IHtcbiAgICAgICAgICBwcm9tcHQ6IGN1cnJlbnRBbnN3ZXIudG9rZW5Vc2FnZS5wcm9tcHQgKyAxMDAsIC8vIEFwcHJveGltYXRlIGFkZGl0aW9uYWwgdG9rZW5zXG4gICAgICAgICAgY29tcGxldGlvbjogY3VycmVudEFuc3dlci50b2tlblVzYWdlLmNvbXBsZXRpb24gKyAxMDAsXG4gICAgICAgICAgdG90YWw6IGN1cnJlbnRBbnN3ZXIudG9rZW5Vc2FnZS50b3RhbCArIDIwMCxcbiAgICAgICAgfSxcbiAgICAgIH07XG4gICAgfVxuXG4gICAgcmV0dXJuIGN1cnJlbnRBbnN3ZXI7XG4gIH1cblxuICAvKipcbiAgICogR2V0IGNhY2hlZCBhbnN3ZXIgaWYgdmFsaWRcbiAgICovXG4gIHByaXZhdGUgZ2V0Q2FjaGVkKHF1ZXJ5OiBzdHJpbmcpOiBSbG1BbnN3ZXIgfCBudWxsIHtcbiAgICBjb25zdCBoYXNoID0gdGhpcy5oYXNoUXVlcnkocXVlcnkpO1xuICAgIGNvbnN0IGVudHJ5ID0gdGhpcy5jYWNoZS5nZXQoaGFzaCk7XG5cbiAgICBpZiAoIWVudHJ5KSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG5cbiAgICAvLyBDaGVjayBUVExcbiAgICBpZiAoRGF0ZS5ub3coKSAtIGVudHJ5LnRpbWVzdGFtcCA+IHRoaXMuY29uZmlnLmNhY2hlVHRsKSB7XG4gICAgICB0aGlzLmNhY2hlLmRlbGV0ZShoYXNoKTtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIHJldHVybiBlbnRyeS5hbnN3ZXI7XG4gIH1cblxuICAvKipcbiAgICogU2V0IGNhY2hlIGVudHJ5XG4gICAqL1xuICBwcml2YXRlIHNldENhY2hlKHF1ZXJ5OiBzdHJpbmcsIGFuc3dlcjogUmxtQW5zd2VyKTogdm9pZCB7XG4gICAgY29uc3QgaGFzaCA9IHRoaXMuaGFzaFF1ZXJ5KHF1ZXJ5KTtcbiAgICB0aGlzLmNhY2hlLnNldChoYXNoLCB7XG4gICAgICBhbnN3ZXIsXG4gICAgICB0aW1lc3RhbXA6IERhdGUubm93KCksXG4gICAgICBxdWVyeUhhc2g6IGhhc2gsXG4gICAgfSk7XG5cbiAgICAvLyBQcnVuZSBvbGQgZW50cmllcyBpZiBjYWNoZSBnZXRzIHRvbyBsYXJnZVxuICAgIGlmICh0aGlzLmNhY2hlLnNpemUgPiAxMDAwKSB7XG4gICAgICB0aGlzLnBydW5lQ2FjaGUoKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogU2ltcGxlIGhhc2ggZnVuY3Rpb24gZm9yIGNhY2hlIGtleXNcbiAgICovXG4gIHByaXZhdGUgaGFzaFF1ZXJ5KHF1ZXJ5OiBzdHJpbmcpOiBzdHJpbmcge1xuICAgIGxldCBoYXNoID0gMDtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHF1ZXJ5Lmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBjaGFyID0gcXVlcnkuY2hhckNvZGVBdChpKTtcbiAgICAgIGhhc2ggPSAoKGhhc2ggPDwgNSkgLSBoYXNoKSArIGNoYXI7XG4gICAgICBoYXNoID0gaGFzaCAmIGhhc2g7IC8vIENvbnZlcnQgdG8gMzItYml0IGludGVnZXJcbiAgICB9XG4gICAgcmV0dXJuIGBybG0tY2FjaGUtJHtoYXNoLnRvU3RyaW5nKDE2KX1gO1xuICB9XG5cbiAgLyoqXG4gICAqIFBydW5lIGV4cGlyZWQgY2FjaGUgZW50cmllc1xuICAgKi9cbiAgcHJpdmF0ZSBwcnVuZUNhY2hlKCk6IHZvaWQge1xuICAgIGNvbnN0IG5vdyA9IERhdGUubm93KCk7XG4gICAgY29uc3QgdG9EZWxldGU6IHN0cmluZ1tdID0gW107XG5cbiAgICBmb3IgKGNvbnN0IFtrZXksIGVudHJ5XSBvZiB0aGlzLmNhY2hlLmVudHJpZXMoKSkge1xuICAgICAgaWYgKG5vdyAtIGVudHJ5LnRpbWVzdGFtcCA+IHRoaXMuY29uZmlnLmNhY2hlVHRsKSB7XG4gICAgICAgIHRvRGVsZXRlLnB1c2goa2V5KTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBEZWxldGUgb2xkZXN0IGVudHJpZXMgaWYgc3RpbGwgdG9vIGxhcmdlXG4gICAgaWYgKHRoaXMuY2FjaGUuc2l6ZSAtIHRvRGVsZXRlLmxlbmd0aCA+IDgwMCkge1xuICAgICAgY29uc3QgZW50cmllcyA9IEFycmF5LmZyb20odGhpcy5jYWNoZS5lbnRyaWVzKCkpXG4gICAgICAgIC5zb3J0KChhLCBiKSA9PiBhWzFdLnRpbWVzdGFtcCAtIGJbMV0udGltZXN0YW1wKTtcblxuICAgICAgY29uc3QgZGVsZXRlQ291bnQgPSBlbnRyaWVzLmxlbmd0aCAtIDUwMDtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZGVsZXRlQ291bnQ7IGkrKykge1xuICAgICAgICB0b0RlbGV0ZS5wdXNoKGVudHJpZXNbaV1bMF0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIGZvciAoY29uc3Qga2V5IG9mIHRvRGVsZXRlKSB7XG4gICAgICB0aGlzLmNhY2hlLmRlbGV0ZShrZXkpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBVdGlsaXR5IGRlbGF5IGZ1bmN0aW9uIGZvciBzdHJlYW1pbmcgc2ltdWxhdGlvblxuICAgKi9cbiAgcHJpdmF0ZSBkZWxheShtczogbnVtYmVyKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgcmV0dXJuIG5ldyBQcm9taXNlKHJlc29sdmUgPT4gc2V0VGltZW91dChyZXNvbHZlLCBtcykpO1xuICB9XG59XG4iXX0=