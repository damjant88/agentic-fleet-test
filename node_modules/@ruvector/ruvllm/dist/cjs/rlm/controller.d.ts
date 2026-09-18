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
import { RlmConfig, RlmAnswer, MemorySpan, StreamToken } from './types';
import { RuvLLM } from '../engine';
/**
 * RlmController - Recursive Retrieval Language Model Controller
 *
 * Orchestrates retrieval-augmented generation with recursive sub-query
 * decomposition, memory search, and optional self-reflection.
 */
export declare class RlmController {
    private config;
    private cache;
    private engine;
    private memoryIdCounter;
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
    constructor(config?: RlmConfig, engine?: RuvLLM);
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
    query(input: string): Promise<RlmAnswer>;
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
    queryStream(input: string): AsyncGenerator<StreamToken>;
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
    addMemory(text: string, metadata?: Record<string, unknown>): Promise<string>;
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
    searchMemory(query: string, topK?: number): Promise<MemorySpan[]>;
    /**
     * Clear the response cache
     *
     * @example
     * ```typescript
     * rlm.clearCache();
     * console.log('Cache cleared');
     * ```
     */
    clearCache(): void;
    /**
     * Get current cache statistics
     *
     * @returns Object with cache size and hit rate info
     */
    getCacheStats(): {
        size: number;
        entries: number;
    };
    /**
     * Update configuration at runtime
     *
     * @param config - Partial configuration to merge
     */
    updateConfig(config: Partial<RlmConfig>): void;
    /**
     * Get current configuration
     */
    getConfig(): Required<RlmConfig>;
    /**
     * Generate sub-queries for complex questions
     */
    private generateSubQueries;
    /**
     * Decompose a complex query into simpler parts
     */
    private decomposeQuery;
    /**
     * Build context string from sources and sub-queries
     */
    private buildContext;
    /**
     * Build the full prompt with context
     */
    private buildPrompt;
    /**
     * Get generation config based on RLM settings
     */
    private getGenerationConfig;
    /**
     * Estimate token usage
     */
    private estimateTokenUsage;
    /**
     * Calculate quality score based on sources and confidence
     */
    private calculateQualityScore;
    /**
     * Apply self-reflection to improve answer
     */
    private applyReflection;
    /**
     * Get cached answer if valid
     */
    private getCached;
    /**
     * Set cache entry
     */
    private setCache;
    /**
     * Simple hash function for cache keys
     */
    private hashQuery;
    /**
     * Prune expired cache entries
     */
    private pruneCache;
    /**
     * Utility delay function for streaming simulation
     */
    private delay;
}
//# sourceMappingURL=controller.d.ts.map