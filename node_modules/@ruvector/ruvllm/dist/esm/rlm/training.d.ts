/**
 * RLM (Recursive Learning Machine) Training Module
 *
 * Provides training capabilities for RuvLTRA models on RLM task routing
 * and decomposition, including query decomposition, answer synthesis,
 * and agent routing optimization.
 *
 * @module rlm/training
 */
/**
 * Strategy for decomposing a complex query
 */
export type DecompositionStrategy = 'sequential' | 'parallel' | 'hierarchical' | 'dag-based' | 'iterative' | 'none';
/**
 * A sub-query in the decomposition
 */
export interface SubQuery {
    /** Unique identifier within the decomposition */
    id: number;
    /** The sub-query text */
    query: string;
    /** Expected output type (e.g., "code", "analysis", "data") */
    expectedType: string;
    /** Dependencies (IDs of sub-queries that must complete first) */
    dependencies: number[];
    /** Recommended agent type for this sub-query */
    recommendedAgent?: string;
    /** Estimated complexity (0.0-1.0) */
    complexity: number;
    /** Optional context from parent query */
    context?: string;
}
/**
 * Decomposition of a complex query into sub-queries
 */
export interface QueryDecomposition {
    /** Sub-queries in execution order */
    subQueries: SubQuery[];
    /** Decomposition strategy used */
    strategy: DecompositionStrategy;
    /** Reasoning for this decomposition */
    rationale: string;
    /** Total estimated complexity */
    totalComplexity: number;
    /** Whether decomposition was successful */
    success: boolean;
    /** Error message if decomposition failed */
    error?: string;
}
/**
 * Answer to a sub-query
 */
export interface SubAnswer {
    /** ID of the sub-query this answers */
    subQueryId: number;
    /** The answer content */
    content: string;
    /** Confidence in this answer (0.0-1.0) */
    confidence: number;
    /** Agent that produced this answer */
    agent: string;
    /** Latency in milliseconds */
    latencyMs: number;
    /** Quality score (0.0-1.0) */
    quality: number;
    /** Whether this answer was successful */
    success: boolean;
    /** Error message if failed */
    error?: string;
    /** Intermediate reasoning/chain-of-thought */
    reasoning?: string;
}
/**
 * Metadata about the RLM execution trajectory
 */
export interface RlmTrajectoryMetadata {
    /** Session ID */
    sessionId?: string;
    /** User ID */
    userId?: string;
    /** Total latency in milliseconds */
    totalLatencyMs: number;
    /** Number of retries */
    retries: number;
    /** Maximum parallel branches executed */
    maxParallelism: number;
    /** Models used during execution */
    modelsUsed: string[];
    /** Agents invoked */
    agentsInvoked: string[];
    /** Tools used */
    toolsUsed: string[];
    /** Custom attributes */
    attributes: Record<string, string>;
}
/**
 * A complete RLM training example
 */
export interface RlmTrainingExample {
    /** Unique identifier */
    id: string;
    /** Original complex query */
    query: string;
    /** Query embedding (optional) */
    queryEmbedding?: number[];
    /** How the query was decomposed */
    decomposition: QueryDecomposition;
    /** Answers to each sub-query */
    subAnswers: SubAnswer[];
    /** Final synthesized answer */
    finalAnswer: string;
    /** Final answer embedding (optional) */
    finalEmbedding?: number[];
    /** Overall quality score (0.0-1.0) */
    qualityScore: number;
    /** Execution trajectory metadata */
    trajectory: RlmTrajectoryMetadata;
    /** Whether this example was successful */
    success: boolean;
    /** Lessons learned from this example */
    lessons: string[];
    /** Source of this example */
    source: string;
}
/**
 * A contrastive pair for agent routing training
 */
export interface ContrastivePair {
    /** Anchor query */
    anchor: string;
    /** Anchor embedding (optional) */
    anchorEmbedding?: number[];
    /** Positive agent (correct routing) */
    positiveAgent: string;
    /** Negative agent (incorrect routing) */
    negativeAgent: string;
    /** Whether this is a hard negative */
    isHardNegative: boolean;
    /** Quality score of the anchor example */
    quality: number;
    /** Source example ID */
    sourceId: string;
}
/**
 * Configuration for RLM training
 */
export interface RlmTrainingConfig {
    /** Learning rate for decomposition training */
    decompositionLr: number;
    /** Learning rate for synthesis training */
    synthesisLr: number;
    /** Learning rate for contrastive fine-tuning */
    contrastiveLr: number;
    /** Batch size */
    batchSize: number;
    /** Number of epochs */
    epochs: number;
    /** Contrastive margin for triplet loss */
    contrastiveMargin: number;
    /** Temperature for InfoNCE loss */
    infonceTemperature: number;
    /** Weight for decomposition loss */
    decompositionWeight: number;
    /** Weight for synthesis loss */
    synthesisWeight: number;
    /** Weight for routing loss */
    routingWeight: number;
    /** Minimum quality for updates */
    qualityThreshold: number;
    /** Evaluation interval (epochs) */
    evaluationInterval: number;
    /** Warmup steps */
    warmupSteps: number;
    /** Early stopping patience */
    earlyStoppingPatience: number;
    /** Validation split ratio */
    validationSplit: number;
    /** Random seed */
    seed: number;
}
/**
 * Training result for a phase
 */
export interface TrainingResult {
    /** Training phase name */
    phase: string;
    /** Epochs completed */
    epochsCompleted: number;
    /** Total steps */
    totalSteps: number;
    /** Final training loss */
    finalLoss: number;
    /** Best validation loss */
    bestValLoss: number;
    /** Best epoch */
    bestEpoch: number;
    /** Final accuracy (for classification tasks) */
    accuracy: number;
    /** Loss history per epoch */
    lossHistory: number[];
    /** Validation loss history */
    valLossHistory: number[];
    /** Training duration in milliseconds */
    durationMs: number;
    /** Whether early stopping was triggered */
    earlyStopped: boolean;
}
/**
 * Evaluation result for the trained model
 */
export interface EvaluationResult {
    /** Decomposition accuracy */
    decompositionAccuracy: number;
    /** Synthesis quality */
    synthesisQuality: number;
    /** Routing accuracy */
    routingAccuracy: number;
    /** Hard negative accuracy */
    hardNegativeAccuracy: number;
    /** Average latency in ms */
    avgLatencyMs: number;
    /** Total examples evaluated */
    totalExamples: number;
    /** Per-agent accuracy */
    perAgentAccuracy: Record<string, number>;
}
/**
 * Default RLM training configuration
 */
export declare const DEFAULT_RLM_CONFIG: RlmTrainingConfig;
/**
 * Fast training configuration
 */
export declare const FAST_RLM_CONFIG: RlmTrainingConfig;
/**
 * Thorough training configuration
 */
export declare const THOROUGH_RLM_CONFIG: RlmTrainingConfig;
/**
 * Routing-focused training configuration
 */
export declare const ROUTING_FOCUSED_CONFIG: RlmTrainingConfig;
/**
 * Agent types with descriptions and keywords
 */
export declare const AGENT_DEFINITIONS: Record<string, {
    description: string;
    keywords: string[];
}>;
/**
 * Hard negative pairs (confusable agent combinations)
 */
export declare const HARD_NEGATIVE_PAIRS: [string, string][];
/**
 * RLM Trainer for RuvLTRA models
 *
 * Provides training capabilities for decomposition, synthesis, and routing tasks.
 */
export declare class RlmTrainer {
    private config;
    private currentEpoch;
    private currentStep;
    private bestValLoss;
    private patienceCounter;
    private lossHistory;
    private valLossHistory;
    /**
     * Create a new RLM trainer
     */
    constructor(config?: Partial<RlmTrainingConfig>);
    /**
     * Train on decomposition task
     *
     * Learns to break complex queries into manageable sub-queries.
     */
    trainDecomposition(dataset: RlmTrainingExample[]): Promise<TrainingResult>;
    /**
     * Train on synthesis task
     *
     * Learns to combine sub-answers into coherent final responses.
     */
    trainSynthesis(dataset: RlmTrainingExample[]): Promise<TrainingResult>;
    /**
     * Contrastive fine-tuning for agent routing
     *
     * Uses triplet loss and InfoNCE to improve routing accuracy.
     */
    trainContrastive(pairs: ContrastivePair[]): Promise<TrainingResult>;
    /**
     * Evaluate trained model on test set
     */
    evaluate(testSet: RlmTrainingExample[]): Promise<EvaluationResult>;
    /**
     * Generate contrastive pairs from dataset
     */
    generateContrastivePairs(dataset: RlmTrainingExample[], hardNegativeRatio?: number): ContrastivePair[];
    private resetState;
    private splitDataset;
    private splitPairs;
    private createBatches;
    private createPairBatches;
    private shuffle;
    private trainDecompositionBatch;
    private trainSynthesisBatch;
    private trainContrastiveBatch;
    private validateDecomposition;
    private validateSynthesis;
    private validateContrastive;
    private computeTripletLoss;
    private computeInfoNCELoss;
    private agentDistance;
    private predictAgent;
    private isHardNegative;
    private findBestEpoch;
}
/**
 * Create an RLM trainer with default configuration
 */
export declare function createRlmTrainer(config?: Partial<RlmTrainingConfig>): RlmTrainer;
/**
 * Create an empty RLM training example
 */
export declare function createEmptyExample(query: string): RlmTrainingExample;
/**
 * Create a sub-query
 */
export declare function createSubQuery(id: number, query: string, options?: Partial<SubQuery>): SubQuery;
/**
 * Create a sub-answer
 */
export declare function createSubAnswer(subQueryId: number, content: string, agent: string, options?: Partial<SubAnswer>): SubAnswer;
export default RlmTrainer;
//# sourceMappingURL=training.d.ts.map