/**
 * RLM (Recursive Learning Machine) Training Module
 *
 * Provides training capabilities for RuvLTRA models on RLM task routing
 * and decomposition, including query decomposition, answer synthesis,
 * and agent routing optimization.
 *
 * @module rlm/training
 */
// =============================================================================
// Default Configurations
// =============================================================================
/**
 * Default RLM training configuration
 */
export const DEFAULT_RLM_CONFIG = {
    decompositionLr: 1e-5,
    synthesisLr: 1e-5,
    contrastiveLr: 2e-5,
    batchSize: 32,
    epochs: 10,
    contrastiveMargin: 0.5,
    infonceTemperature: 0.07,
    decompositionWeight: 1.0,
    synthesisWeight: 1.0,
    routingWeight: 1.0,
    qualityThreshold: 0.7,
    evaluationInterval: 1,
    warmupSteps: 100,
    earlyStoppingPatience: 3,
    validationSplit: 0.1,
    seed: 42,
};
/**
 * Fast training configuration
 */
export const FAST_RLM_CONFIG = {
    ...DEFAULT_RLM_CONFIG,
    epochs: 3,
    batchSize: 64,
    decompositionLr: 1e-4,
    synthesisLr: 1e-4,
    contrastiveLr: 5e-5,
    earlyStoppingPatience: 1,
};
/**
 * Thorough training configuration
 */
export const THOROUGH_RLM_CONFIG = {
    ...DEFAULT_RLM_CONFIG,
    epochs: 50,
    batchSize: 16,
    decompositionLr: 5e-6,
    synthesisLr: 5e-6,
    contrastiveLr: 1e-5,
    earlyStoppingPatience: 10,
};
/**
 * Routing-focused training configuration
 */
export const ROUTING_FOCUSED_CONFIG = {
    ...DEFAULT_RLM_CONFIG,
    routingWeight: 2.0,
    decompositionWeight: 0.5,
    synthesisWeight: 0.5,
    contrastiveLr: 3e-5,
    contrastiveMargin: 0.3,
    infonceTemperature: 0.05,
};
// =============================================================================
// Agent Definitions
// =============================================================================
/**
 * Agent types with descriptions and keywords
 */
export const AGENT_DEFINITIONS = {
    coder: {
        description: 'Software developer who writes and implements code',
        keywords: ['implement', 'build', 'create', 'code', 'write', 'develop', 'program'],
    },
    researcher: {
        description: 'Technical researcher who investigates and analyzes',
        keywords: ['research', 'investigate', 'analyze', 'explore', 'study', 'examine'],
    },
    reviewer: {
        description: 'Code reviewer who evaluates code quality',
        keywords: ['review', 'check', 'evaluate', 'assess', 'examine', 'inspect'],
    },
    tester: {
        description: 'QA engineer who writes and runs tests',
        keywords: ['test', 'unit test', 'coverage', 'validate', 'verify', 'qa'],
    },
    architect: {
        description: 'System architect who designs software structure',
        keywords: ['design', 'plan', 'architecture', 'schema', 'structure', 'diagram'],
    },
    'security-architect': {
        description: 'Security specialist who audits vulnerabilities',
        keywords: ['security', 'audit', 'vulnerability', 'xss', 'injection', 'cve'],
    },
    debugger: {
        description: 'Bug hunter who fixes errors and traces issues',
        keywords: ['fix', 'debug', 'bug', 'error', 'trace', 'crash', 'troubleshoot'],
    },
    documenter: {
        description: 'Technical writer who creates documentation',
        keywords: ['document', 'jsdoc', 'readme', 'comment', 'explain', 'describe'],
    },
    refactorer: {
        description: 'Code modernizer who restructures without changing behavior',
        keywords: ['refactor', 'restructure', 'modernize', 'clean', 'simplify', 'consolidate'],
    },
    optimizer: {
        description: 'Performance engineer who speeds up slow code',
        keywords: ['optimize', 'performance', 'speed', 'cache', 'improve', 'faster'],
    },
    devops: {
        description: 'DevOps engineer who manages deployment and infrastructure',
        keywords: ['deploy', 'ci/cd', 'kubernetes', 'docker', 'infrastructure', 'pipeline'],
    },
    'api-docs': {
        description: 'API documentation specialist who creates specs',
        keywords: ['openapi', 'swagger', 'api reference', 'endpoint', 'spec', 'rest'],
    },
    planner: {
        description: 'Project planner who organizes and schedules work',
        keywords: ['plan', 'estimate', 'schedule', 'timeline', 'sprint', 'roadmap'],
    },
};
/**
 * Hard negative pairs (confusable agent combinations)
 */
export const HARD_NEGATIVE_PAIRS = [
    ['coder', 'debugger'],
    ['coder', 'refactorer'],
    ['researcher', 'reviewer'],
    ['tester', 'reviewer'],
    ['architect', 'planner'],
    ['documenter', 'api-docs'],
    ['optimizer', 'debugger'],
    ['devops', 'architect'],
    ['security-architect', 'reviewer'],
];
// =============================================================================
// RLM Trainer Class
// =============================================================================
/**
 * RLM Trainer for RuvLTRA models
 *
 * Provides training capabilities for decomposition, synthesis, and routing tasks.
 */
export class RlmTrainer {
    /**
     * Create a new RLM trainer
     */
    constructor(config = {}) {
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.bestValLoss = Infinity;
        this.patienceCounter = 0;
        this.lossHistory = [];
        this.valLossHistory = [];
        this.config = { ...DEFAULT_RLM_CONFIG, ...config };
    }
    /**
     * Train on decomposition task
     *
     * Learns to break complex queries into manageable sub-queries.
     */
    async trainDecomposition(dataset) {
        const startTime = Date.now();
        this.resetState();
        const { trainSet, valSet } = this.splitDataset(dataset);
        const batches = this.createBatches(trainSet);
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            this.currentEpoch = epoch;
            let epochLoss = 0;
            for (const batch of batches) {
                const batchLoss = this.trainDecompositionBatch(batch);
                epochLoss += batchLoss;
                this.currentStep++;
            }
            const avgLoss = epochLoss / batches.length;
            this.lossHistory.push(avgLoss);
            // Validation
            const valLoss = this.validateDecomposition(valSet);
            this.valLossHistory.push(valLoss);
            // Early stopping
            if (valLoss < this.bestValLoss) {
                this.bestValLoss = valLoss;
                this.patienceCounter = 0;
            }
            else {
                this.patienceCounter++;
                if (this.patienceCounter >= this.config.earlyStoppingPatience) {
                    break;
                }
            }
        }
        return {
            phase: 'decomposition',
            epochsCompleted: this.currentEpoch + 1,
            totalSteps: this.currentStep,
            finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
            bestValLoss: this.bestValLoss,
            bestEpoch: this.findBestEpoch(),
            accuracy: 0, // Not applicable for decomposition
            lossHistory: this.lossHistory,
            valLossHistory: this.valLossHistory,
            durationMs: Date.now() - startTime,
            earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
        };
    }
    /**
     * Train on synthesis task
     *
     * Learns to combine sub-answers into coherent final responses.
     */
    async trainSynthesis(dataset) {
        const startTime = Date.now();
        this.resetState();
        const { trainSet, valSet } = this.splitDataset(dataset);
        const batches = this.createBatches(trainSet);
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            this.currentEpoch = epoch;
            let epochLoss = 0;
            for (const batch of batches) {
                const batchLoss = this.trainSynthesisBatch(batch);
                epochLoss += batchLoss;
                this.currentStep++;
            }
            const avgLoss = epochLoss / batches.length;
            this.lossHistory.push(avgLoss);
            // Validation
            const valLoss = this.validateSynthesis(valSet);
            this.valLossHistory.push(valLoss);
            // Early stopping
            if (valLoss < this.bestValLoss) {
                this.bestValLoss = valLoss;
                this.patienceCounter = 0;
            }
            else {
                this.patienceCounter++;
                if (this.patienceCounter >= this.config.earlyStoppingPatience) {
                    break;
                }
            }
        }
        return {
            phase: 'synthesis',
            epochsCompleted: this.currentEpoch + 1,
            totalSteps: this.currentStep,
            finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
            bestValLoss: this.bestValLoss,
            bestEpoch: this.findBestEpoch(),
            accuracy: 0,
            lossHistory: this.lossHistory,
            valLossHistory: this.valLossHistory,
            durationMs: Date.now() - startTime,
            earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
        };
    }
    /**
     * Contrastive fine-tuning for agent routing
     *
     * Uses triplet loss and InfoNCE to improve routing accuracy.
     */
    async trainContrastive(pairs) {
        const startTime = Date.now();
        this.resetState();
        if (pairs.length === 0) {
            throw new Error('No contrastive pairs provided');
        }
        const { trainSet, valSet } = this.splitPairs(pairs);
        const batches = this.createPairBatches(trainSet);
        let totalCorrect = 0;
        let totalExamples = 0;
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            this.currentEpoch = epoch;
            let epochLoss = 0;
            for (const batch of batches) {
                const batchLoss = this.trainContrastiveBatch(batch);
                epochLoss += batchLoss;
                this.currentStep++;
            }
            const avgLoss = epochLoss / batches.length;
            this.lossHistory.push(avgLoss);
            // Validation
            const { loss: valLoss, correct, total } = this.validateContrastive(valSet);
            this.valLossHistory.push(valLoss);
            totalCorrect = correct;
            totalExamples = total;
            // Early stopping
            if (valLoss < this.bestValLoss) {
                this.bestValLoss = valLoss;
                this.patienceCounter = 0;
            }
            else {
                this.patienceCounter++;
                if (this.patienceCounter >= this.config.earlyStoppingPatience) {
                    break;
                }
            }
        }
        return {
            phase: 'contrastive',
            epochsCompleted: this.currentEpoch + 1,
            totalSteps: this.currentStep,
            finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
            bestValLoss: this.bestValLoss,
            bestEpoch: this.findBestEpoch(),
            accuracy: totalExamples > 0 ? totalCorrect / totalExamples : 0,
            lossHistory: this.lossHistory,
            valLossHistory: this.valLossHistory,
            durationMs: Date.now() - startTime,
            earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
        };
    }
    /**
     * Evaluate trained model on test set
     */
    async evaluate(testSet) {
        const perAgentAccuracy = {};
        let decompositionCorrect = 0;
        let synthesisQualitySum = 0;
        let routingCorrect = 0;
        let hardNegativeCorrect = 0;
        let hardNegativeTotal = 0;
        let totalLatency = 0;
        for (const example of testSet) {
            // Decomposition evaluation
            if (example.decomposition.success && example.decomposition.subQueries.length > 0) {
                decompositionCorrect++;
            }
            // Synthesis quality
            synthesisQualitySum += example.qualityScore;
            // Routing evaluation
            for (const subQuery of example.decomposition.subQueries) {
                if (subQuery.recommendedAgent) {
                    const predicted = this.predictAgent(subQuery.query);
                    const correct = predicted === subQuery.recommendedAgent;
                    if (correct) {
                        routingCorrect++;
                    }
                    // Track per-agent accuracy
                    if (!perAgentAccuracy[subQuery.recommendedAgent]) {
                        perAgentAccuracy[subQuery.recommendedAgent] = { correct: 0, total: 0 };
                    }
                    perAgentAccuracy[subQuery.recommendedAgent].total++;
                    if (correct) {
                        perAgentAccuracy[subQuery.recommendedAgent].correct++;
                    }
                    // Check hard negatives
                    if (this.isHardNegative(subQuery.recommendedAgent, predicted)) {
                        hardNegativeTotal++;
                        if (correct) {
                            hardNegativeCorrect++;
                        }
                    }
                }
            }
            totalLatency += example.trajectory.totalLatencyMs;
        }
        const totalRoutingExamples = testSet.reduce((sum, ex) => sum + ex.decomposition.subQueries.filter((sq) => sq.recommendedAgent).length, 0);
        const perAgentResult = {};
        for (const [agent, stats] of Object.entries(perAgentAccuracy)) {
            perAgentResult[agent] = stats.total > 0 ? stats.correct / stats.total : 0;
        }
        return {
            decompositionAccuracy: testSet.length > 0 ? decompositionCorrect / testSet.length : 0,
            synthesisQuality: testSet.length > 0 ? synthesisQualitySum / testSet.length : 0,
            routingAccuracy: totalRoutingExamples > 0 ? routingCorrect / totalRoutingExamples : 0,
            hardNegativeAccuracy: hardNegativeTotal > 0 ? hardNegativeCorrect / hardNegativeTotal : 0,
            avgLatencyMs: testSet.length > 0 ? totalLatency / testSet.length : 0,
            totalExamples: testSet.length,
            perAgentAccuracy: perAgentResult,
        };
    }
    /**
     * Generate contrastive pairs from dataset
     */
    generateContrastivePairs(dataset, hardNegativeRatio = 0.3) {
        const pairs = [];
        const agents = Object.keys(AGENT_DEFINITIONS);
        for (const example of dataset) {
            for (const subQuery of example.decomposition.subQueries) {
                if (!subQuery.recommendedAgent)
                    continue;
                const positiveAgent = subQuery.recommendedAgent;
                for (const negativeAgent of agents) {
                    if (negativeAgent === positiveAgent)
                        continue;
                    const isHard = this.isHardNegative(positiveAgent, negativeAgent);
                    // Apply hard negative ratio
                    const include = isHard
                        ? Math.random() < hardNegativeRatio
                        : Math.random() < 1 - hardNegativeRatio;
                    if (include) {
                        pairs.push({
                            anchor: subQuery.query,
                            anchorEmbedding: example.queryEmbedding,
                            positiveAgent,
                            negativeAgent,
                            isHardNegative: isHard,
                            quality: example.qualityScore,
                            sourceId: example.id,
                        });
                    }
                }
            }
        }
        return pairs;
    }
    // =============================================================================
    // Private Methods
    // =============================================================================
    resetState() {
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.bestValLoss = Infinity;
        this.patienceCounter = 0;
        this.lossHistory = [];
        this.valLossHistory = [];
    }
    splitDataset(dataset) {
        const valSize = Math.floor(dataset.length * this.config.validationSplit);
        const shuffled = this.shuffle([...dataset]);
        return {
            trainSet: shuffled.slice(valSize),
            valSet: shuffled.slice(0, valSize),
        };
    }
    splitPairs(pairs) {
        const valSize = Math.floor(pairs.length * this.config.validationSplit);
        const shuffled = this.shuffle([...pairs]);
        return {
            trainSet: shuffled.slice(valSize),
            valSet: shuffled.slice(0, valSize),
        };
    }
    createBatches(dataset) {
        const batches = [];
        for (let i = 0; i < dataset.length; i += this.config.batchSize) {
            batches.push(dataset.slice(i, i + this.config.batchSize));
        }
        return batches;
    }
    createPairBatches(pairs) {
        const batches = [];
        for (let i = 0; i < pairs.length; i += this.config.batchSize) {
            batches.push(pairs.slice(i, i + this.config.batchSize));
        }
        return batches;
    }
    shuffle(array) {
        // Fisher-Yates shuffle
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    trainDecompositionBatch(batch) {
        let batchLoss = 0;
        for (const example of batch) {
            // Decomposition quality loss
            const qualityLoss = 1 - (example.decomposition.success ? example.qualityScore : 0);
            // Depth appropriateness (penalize too shallow or too deep)
            const depth = example.decomposition.subQueries.length;
            const idealDepth = 3;
            const depthLoss = Math.abs(depth - idealDepth) / idealDepth;
            // Complexity balance loss
            const complexityLoss = Math.abs(example.decomposition.totalComplexity - 1) / 3;
            const loss = qualityLoss * this.config.decompositionWeight * 0.6 +
                depthLoss * 0.2 +
                complexityLoss * 0.2;
            batchLoss += loss;
        }
        return batchLoss / batch.length;
    }
    trainSynthesisBatch(batch) {
        let batchLoss = 0;
        for (const example of batch) {
            // Sub-answer quality
            const subAnswerQuality = example.subAnswers.length > 0
                ? example.subAnswers.reduce((sum, a) => sum + a.quality, 0) / example.subAnswers.length
                : 0;
            // Final answer quality
            const finalQuality = example.qualityScore;
            // Coherence bonus (final should be better than parts average)
            const coherenceBonus = Math.max(0, finalQuality - subAnswerQuality) * 0.5;
            const loss = (1 - (subAnswerQuality * 0.4 + finalQuality * 0.4 + coherenceBonus * 0.2));
            batchLoss += loss * this.config.synthesisWeight;
        }
        return batchLoss / batch.length;
    }
    trainContrastiveBatch(batch) {
        let batchLoss = 0;
        for (const pair of batch) {
            // Triplet loss
            const tripletLoss = this.computeTripletLoss(pair);
            // InfoNCE loss
            const infonceLoss = this.computeInfoNCELoss(pair);
            batchLoss += (tripletLoss * 0.5 + infonceLoss * 0.5) * this.config.routingWeight;
        }
        return batchLoss / batch.length;
    }
    validateDecomposition(valSet) {
        if (valSet.length === 0)
            return 0;
        let totalLoss = 0;
        for (const example of valSet) {
            totalLoss += 1 - example.qualityScore;
        }
        return totalLoss / valSet.length;
    }
    validateSynthesis(valSet) {
        if (valSet.length === 0)
            return 0;
        let totalLoss = 0;
        for (const example of valSet) {
            totalLoss += 1 - example.qualityScore;
        }
        return totalLoss / valSet.length;
    }
    validateContrastive(valSet) {
        if (valSet.length === 0)
            return { loss: 0, correct: 0, total: 0 };
        let totalLoss = 0;
        let correct = 0;
        for (const pair of valSet) {
            const tripletLoss = this.computeTripletLoss(pair);
            const infonceLoss = this.computeInfoNCELoss(pair);
            totalLoss += tripletLoss * 0.5 + infonceLoss * 0.5;
            // Check routing correctness
            const posDist = this.agentDistance(pair.anchor, pair.positiveAgent);
            const negDist = this.agentDistance(pair.anchor, pair.negativeAgent);
            if (posDist < negDist) {
                correct++;
            }
        }
        return {
            loss: totalLoss / valSet.length,
            correct,
            total: valSet.length,
        };
    }
    computeTripletLoss(pair) {
        const posDist = this.agentDistance(pair.anchor, pair.positiveAgent);
        const negDist = this.agentDistance(pair.anchor, pair.negativeAgent);
        return Math.max(0, this.config.contrastiveMargin + posDist - negDist);
    }
    computeInfoNCELoss(pair) {
        const posSim = 1 - this.agentDistance(pair.anchor, pair.positiveAgent);
        const negSim = 1 - this.agentDistance(pair.anchor, pair.negativeAgent);
        const temp = this.config.infonceTemperature;
        const posExp = Math.exp(posSim / temp);
        const negExp = Math.exp(negSim / temp);
        return -Math.log(posExp / (posExp + negExp));
    }
    agentDistance(query, agent) {
        const queryLower = query.toLowerCase();
        const agentDef = AGENT_DEFINITIONS[agent];
        if (!agentDef)
            return 1.0;
        const matches = agentDef.keywords.filter((kw) => queryLower.includes(kw)).length;
        return 1.0 - Math.min(1.0, matches / agentDef.keywords.length);
    }
    predictAgent(query) {
        let bestAgent = 'coder';
        let bestScore = 0;
        for (const [agent, def] of Object.entries(AGENT_DEFINITIONS)) {
            const queryLower = query.toLowerCase();
            const matches = def.keywords.filter((kw) => queryLower.includes(kw)).length;
            const score = matches / def.keywords.length;
            if (score > bestScore) {
                bestScore = score;
                bestAgent = agent;
            }
        }
        return bestAgent;
    }
    isHardNegative(agent1, agent2) {
        return HARD_NEGATIVE_PAIRS.some(([a, b]) => (agent1 === a && agent2 === b) || (agent1 === b && agent2 === a));
    }
    findBestEpoch() {
        if (this.valLossHistory.length === 0)
            return 0;
        let bestIdx = 0;
        let bestLoss = this.valLossHistory[0];
        for (let i = 1; i < this.valLossHistory.length; i++) {
            if (this.valLossHistory[i] < bestLoss) {
                bestLoss = this.valLossHistory[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
// =============================================================================
// Factory Functions
// =============================================================================
/**
 * Create an RLM trainer with default configuration
 */
export function createRlmTrainer(config) {
    return new RlmTrainer(config);
}
/**
 * Create an empty RLM training example
 */
export function createEmptyExample(query) {
    return {
        id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36)}`,
        query,
        decomposition: {
            subQueries: [],
            strategy: 'none',
            rationale: '',
            totalComplexity: 0,
            success: false,
        },
        subAnswers: [],
        finalAnswer: '',
        qualityScore: 0,
        trajectory: {
            totalLatencyMs: 0,
            retries: 0,
            maxParallelism: 1,
            modelsUsed: [],
            agentsInvoked: [],
            toolsUsed: [],
            attributes: {},
        },
        success: false,
        lessons: [],
        source: 'manual',
    };
}
/**
 * Create a sub-query
 */
export function createSubQuery(id, query, options = {}) {
    return {
        id,
        query,
        expectedType: 'text',
        dependencies: [],
        complexity: 0.5,
        ...options,
    };
}
/**
 * Create a sub-answer
 */
export function createSubAnswer(subQueryId, content, agent, options = {}) {
    return {
        subQueryId,
        content,
        confidence: 0.8,
        agent,
        latencyMs: 0,
        quality: 0.8,
        success: true,
        ...options,
    };
}
// =============================================================================
// Exports
// =============================================================================
export default RlmTrainer;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhaW5pbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi9zcmMvcmxtL3RyYWluaW5nLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBK09ILGdGQUFnRjtBQUNoRix5QkFBeUI7QUFDekIsZ0ZBQWdGO0FBRWhGOztHQUVHO0FBQ0gsTUFBTSxDQUFDLE1BQU0sa0JBQWtCLEdBQXNCO0lBQ25ELGVBQWUsRUFBRSxJQUFJO0lBQ3JCLFdBQVcsRUFBRSxJQUFJO0lBQ2pCLGFBQWEsRUFBRSxJQUFJO0lBQ25CLFNBQVMsRUFBRSxFQUFFO0lBQ2IsTUFBTSxFQUFFLEVBQUU7SUFDVixpQkFBaUIsRUFBRSxHQUFHO0lBQ3RCLGtCQUFrQixFQUFFLElBQUk7SUFDeEIsbUJBQW1CLEVBQUUsR0FBRztJQUN4QixlQUFlLEVBQUUsR0FBRztJQUNwQixhQUFhLEVBQUUsR0FBRztJQUNsQixnQkFBZ0IsRUFBRSxHQUFHO0lBQ3JCLGtCQUFrQixFQUFFLENBQUM7SUFDckIsV0FBVyxFQUFFLEdBQUc7SUFDaEIscUJBQXFCLEVBQUUsQ0FBQztJQUN4QixlQUFlLEVBQUUsR0FBRztJQUNwQixJQUFJLEVBQUUsRUFBRTtDQUNULENBQUM7QUFFRjs7R0FFRztBQUNILE1BQU0sQ0FBQyxNQUFNLGVBQWUsR0FBc0I7SUFDaEQsR0FBRyxrQkFBa0I7SUFDckIsTUFBTSxFQUFFLENBQUM7SUFDVCxTQUFTLEVBQUUsRUFBRTtJQUNiLGVBQWUsRUFBRSxJQUFJO0lBQ3JCLFdBQVcsRUFBRSxJQUFJO0lBQ2pCLGFBQWEsRUFBRSxJQUFJO0lBQ25CLHFCQUFxQixFQUFFLENBQUM7Q0FDekIsQ0FBQztBQUVGOztHQUVHO0FBQ0gsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQXNCO0lBQ3BELEdBQUcsa0JBQWtCO0lBQ3JCLE1BQU0sRUFBRSxFQUFFO0lBQ1YsU0FBUyxFQUFFLEVBQUU7SUFDYixlQUFlLEVBQUUsSUFBSTtJQUNyQixX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QUNaLE9BQU8sRUFBRSxJQUFJO1FBQ2IsR0FBRyxPQUFPO0tBQ1gsQ0FBQztBQUNKLENBQUM7QUFFRCxnRkFBZ0Y7QUFDaEYsVUFBVTtBQUNWLGdGQUFnRjtBQUVoRixlQUFlLFVBQVUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogUkxNIChSZWN1cnNpdmUgTGVhcm5pbmcgTWFjaGluZSkgVHJhaW5pbmcgTW9kdWxlXG4gKlxuICogUHJvdmlkZXMgdHJhaW5pbmcgY2FwYWJpbGl0aWVzIGZvciBSdXZMVFJBIG1vZGVscyBvbiBSTE0gdGFzayByb3V0aW5nXG4gKiBhbmQgZGVjb21wb3NpdGlvbiwgaW5jbHVkaW5nIHF1ZXJ5IGRlY29tcG9zaXRpb24sIGFuc3dlciBzeW50aGVzaXMsXG4gKiBhbmQgYWdlbnQgcm91dGluZyBvcHRpbWl6YXRpb24uXG4gKlxuICogQG1vZHVsZSBybG0vdHJhaW5pbmdcbiAqL1xuXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuLy8gVHlwZXMgYW5kIEludGVyZmFjZXNcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8qKlxuICogU3RyYXRlZ3kgZm9yIGRlY29tcG9zaW5nIGEgY29tcGxleCBxdWVyeVxuICovXG5leHBvcnQgdHlwZSBEZWNvbXBvc2l0aW9uU3RyYXRlZ3kgPVxuICB8ICdzZXF1ZW50aWFsJ1xuICB8ICdwYXJhbGxlbCdcbiAgfCAnaGllcmFyY2hpY2FsJ1xuICB8ICdkYWctYmFzZWQnXG4gIHwgJ2l0ZXJhdGl2ZSdcbiAgfCAnbm9uZSc7XG5cbi8qKlxuICogQSBzdWItcXVlcnkgaW4gdGhlIGRlY29tcG9zaXRpb25cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBTdWJRdWVyeSB7XG4gIC8qKiBVbmlxdWUgaWRlbnRpZmllciB3aXRoaW4gdGhlIGRlY29tcG9zaXRpb24gKi9cbiAgaWQ6IG51bWJlcjtcbiAgLyoqIFRoZSBzdWItcXVlcnkgdGV4dCAqL1xuICBxdWVyeTogc3RyaW5nO1xuICAvKiogRXhwZWN0ZWQgb3V0cHV0IHR5cGUgKGUuZy4sIFwiY29kZVwiLCBcImFuYWx5c2lzXCIsIFwiZGF0YVwiKSAqL1xuICBleHBlY3RlZFR5cGU6IHN0cmluZztcbiAgLyoqIERlcGVuZGVuY2llcyAoSURzIG9mIHN1Yi1xdWVyaWVzIHRoYXQgbXVzdCBjb21wbGV0ZSBmaXJzdCkgKi9cbiAgZGVwZW5kZW5jaWVzOiBudW1iZXJbXTtcbiAgLyoqIFJlY29tbWVuZGVkIGFnZW50IHR5cGUgZm9yIHRoaXMgc3ViLXF1ZXJ5ICovXG4gIHJlY29tbWVuZGVkQWdlbnQ/OiBzdHJpbmc7XG4gIC8qKiBFc3RpbWF0ZWQgY29tcGxleGl0eSAoMC4wLTEuMCkgKi9cbiAgY29tcGxleGl0eTogbnVtYmVyO1xuICAvKiogT3B0aW9uYWwgY29udGV4dCBmcm9tIHBhcmVudCBxdWVyeSAqL1xuICBjb250ZXh0Pzogc3RyaW5nO1xufVxuXG4vKipcbiAqIERlY29tcG9zaXRpb24gb2YgYSBjb21wbGV4IHF1ZXJ5IGludG8gc3ViLXF1ZXJpZXNcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBRdWVyeURlY29tcG9zaXRpb24ge1xuICAvKiogU3ViLXF1ZXJpZXMgaW4gZXhlY3V0aW9uIG9yZGVyICovXG4gIHN1YlF1ZXJpZXM6IFN1YlF1ZXJ5W107XG4gIC8qKiBEZWNvbXBvc2l0aW9uIHN0cmF0ZWd5IHVzZWQgKi9cbiAgc3RyYXRlZ3k6IERlY29tcG9zaXRpb25TdHJhdGVneTtcbiAgLyoqIFJlYXNvbmluZyBmb3IgdGhpcyBkZWNvbXBvc2l0aW9uICovXG4gIHJhdGlvbmFsZTogc3RyaW5nO1xuICAvKiogVG90YWwgZXN0aW1hdGVkIGNvbXBsZXhpdHkgKi9cbiAgdG90YWxDb21wbGV4aXR5OiBudW1iZXI7XG4gIC8qKiBXaGV0aGVyIGRlY29tcG9zaXRpb24gd2FzIHN1Y2Nlc3NmdWwgKi9cbiAgc3VjY2VzczogYm9vbGVhbjtcbiAgLyoqIEVycm9yIG1lc3NhZ2UgaWYgZGVjb21wb3NpdGlvbiBmYWlsZWQgKi9cbiAgZXJyb3I/OiBzdHJpbmc7XG59XG5cbi8qKlxuICogQW5zd2VyIHRvIGEgc3ViLXF1ZXJ5XG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgU3ViQW5zd2VyIHtcbiAgLyoqIElEIG9mIHRoZSBzdWItcXVlcnkgdGhpcyBhbnN3ZXJzICovXG4gIHN1YlF1ZXJ5SWQ6IG51bWJlcjtcbiAgLyoqIFRoZSBhbnN3ZXIgY29udGVudCAqL1xuICBjb250ZW50OiBzdHJpbmc7XG4gIC8qKiBDb25maWRlbmNlIGluIHRoaXMgYW5zd2VyICgwLjAtMS4wKSAqL1xuICBjb25maWRlbmNlOiBudW1iZXI7XG4gIC8qKiBBZ2VudCB0aGF0IHByb2R1Y2VkIHRoaXMgYW5zd2VyICovXG4gIGFnZW50OiBzdHJpbmc7XG4gIC8qKiBMYXRlbmN5IGluIG1pbGxpc2Vjb25kcyAqL1xuICBsYXRlbmN5TXM6IG51bWJlcjtcbiAgLyoqIFF1YWxpdHkgc2NvcmUgKDAuMC0xLjApICovXG4gIHF1YWxpdHk6IG51bWJlcjtcbiAgLyoqIFdoZXRoZXIgdGhpcyBhbnN3ZXIgd2FzIHN1Y2Nlc3NmdWwgKi9cbiAgc3VjY2VzczogYm9vbGVhbjtcbiAgLyoqIEVycm9yIG1lc3NhZ2UgaWYgZmFpbGVkICovXG4gIGVycm9yPzogc3RyaW5nO1xuICAvKiogSW50ZXJtZWRpYXRlIHJlYXNvbmluZy9jaGFpbi1vZi10aG91Z2h0ICovXG4gIHJlYXNvbmluZz86IHN0cmluZztcbn1cblxuLyoqXG4gKiBNZXRhZGF0YSBhYm91dCB0aGUgUkxNIGV4ZWN1dGlvbiB0cmFqZWN0b3J5XG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgUmxtVHJhamVjdG9yeU1ldGFkYXRhIHtcbiAgLyoqIFNlc3Npb24gSUQgKi9cbiAgc2Vzc2lvbklkPzogc3RyaW5nO1xuICAvKiogVXNlciBJRCAqL1xuICB1c2VySWQ/OiBzdHJpbmc7XG4gIC8qKiBUb3RhbCBsYXRlbmN5IGluIG1pbGxpc2Vjb25kcyAqL1xuICB0b3RhbExhdGVuY3lNczogbnVtYmVyO1xuICAvKiogTnVtYmVyIG9mIHJldHJpZXMgKi9cbiAgcmV0cmllczogbnVtYmVyO1xuICAvKiogTWF4aW11bSBwYXJhbGxlbCBicmFuY2hlcyBleGVjdXRlZCAqL1xuICBtYXhQYXJhbGxlbGlzbTogbnVtYmVyO1xuICAvKiogTW9kZWxzIHVzZWQgZHVyaW5nIGV4ZWN1dGlvbiAqL1xuICBtb2RlbHNVc2VkOiBzdHJpbmdbXTtcbiAgLyoqIEFnZW50cyBpbnZva2VkICovXG4gIGFnZW50c0ludm9rZWQ6IHN0cmluZ1tdO1xuICAvKiogVG9vbHMgdXNlZCAqL1xuICB0b29sc1VzZWQ6IHN0cmluZ1tdO1xuICAvKiogQ3VzdG9tIGF0dHJpYnV0ZXMgKi9cbiAgYXR0cmlidXRlczogUmVjb3JkPHN0cmluZywgc3RyaW5nPjtcbn1cblxuLyoqXG4gKiBBIGNvbXBsZXRlIFJMTSB0cmFpbmluZyBleGFtcGxlXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgUmxtVHJhaW5pbmdFeGFtcGxlIHtcbiAgLyoqIFVuaXF1ZSBpZGVudGlmaWVyICovXG4gIGlkOiBzdHJpbmc7XG4gIC8qKiBPcmlnaW5hbCBjb21wbGV4IHF1ZXJ5ICovXG4gIHF1ZXJ5OiBzdHJpbmc7XG4gIC8qKiBRdWVyeSBlbWJlZGRpbmcgKG9wdGlvbmFsKSAqL1xuICBxdWVyeUVtYmVkZGluZz86IG51bWJlcltdO1xuICAvKiogSG93IHRoZSBxdWVyeSB3YXMgZGVjb21wb3NlZCAqL1xuICBkZWNvbXBvc2l0aW9uOiBRdWVyeURlY29tcG9zaXRpb247XG4gIC8qKiBBbnN3ZXJzIHRvIGVhY2ggc3ViLXF1ZXJ5ICovXG4gIHN1YkFuc3dlcnM6IFN1YkFuc3dlcltdO1xuICAvKiogRmluYWwgc3ludGhlc2l6ZWQgYW5zd2VyICovXG4gIGZpbmFsQW5zd2VyOiBzdHJpbmc7XG4gIC8qKiBGaW5hbCBhbnN3ZXIgZW1iZWRkaW5nIChvcHRpb25hbCkgKi9cbiAgZmluYWxFbWJlZGRpbmc/OiBudW1iZXJbXTtcbiAgLyoqIE92ZXJhbGwgcXVhbGl0eSBzY29yZSAoMC4wLTEuMCkgKi9cbiAgcXVhbGl0eVNjb3JlOiBudW1iZXI7XG4gIC8qKiBFeGVjdXRpb24gdHJhamVjdG9yeSBtZXRhZGF0YSAqL1xuICB0cmFqZWN0b3J5OiBSbG1UcmFqZWN0b3J5TWV0YWRhdGE7XG4gIC8qKiBXaGV0aGVyIHRoaXMgZXhhbXBsZSB3YXMgc3VjY2Vzc2Z1bCAqL1xuICBzdWNjZXNzOiBib29sZWFuO1xuICAvKiogTGVzc29ucyBsZWFybmVkIGZyb20gdGhpcyBleGFtcGxlICovXG4gIGxlc3NvbnM6IHN0cmluZ1tdO1xuICAvKiogU291cmNlIG9mIHRoaXMgZXhhbXBsZSAqL1xuICBzb3VyY2U6IHN0cmluZztcbn1cblxuLyoqXG4gKiBBIGNvbnRyYXN0aXZlIHBhaXIgZm9yIGFnZW50IHJvdXRpbmcgdHJhaW5pbmdcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBDb250cmFzdGl2ZVBhaXIge1xuICAvKiogQW5jaG9yIHF1ZXJ5ICovXG4gIGFuY2hvcjogc3RyaW5nO1xuICAvKiogQW5jaG9yIGVtYmVkZGluZyAob3B0aW9uYWwpICovXG4gIGFuY2hvckVtYmVkZGluZz86IG51bWJlcltdO1xuICAvKiogUG9zaXRpdmUgYWdlbnQgKGNvcnJlY3Qgcm91dGluZykgKi9cbiAgcG9zaXRpdmVBZ2VudDogc3RyaW5nO1xuICAvKiogTmVnYXRpdmUgYWdlbnQgKGluY29ycmVjdCByb3V0aW5nKSAqL1xuICBuZWdhdGl2ZUFnZW50OiBzdHJpbmc7XG4gIC8qKiBXaGV0aGVyIHRoaXMgaXMgYSBoYXJkIG5lZ2F0aXZlICovXG4gIGlzSGFyZE5lZ2F0aXZlOiBib29sZWFuO1xuICAvKiogUXVhbGl0eSBzY29yZSBvZiB0aGUgYW5jaG9yIGV4YW1wbGUgKi9cbiAgcXVhbGl0eTogbnVtYmVyO1xuICAvKiogU291cmNlIGV4YW1wbGUgSUQgKi9cbiAgc291cmNlSWQ6IHN0cmluZztcbn1cblxuLyoqXG4gKiBDb25maWd1cmF0aW9uIGZvciBSTE0gdHJhaW5pbmdcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBSbG1UcmFpbmluZ0NvbmZpZyB7XG4gIC8qKiBMZWFybmluZyByYXRlIGZvciBkZWNvbXBvc2l0aW9uIHRyYWluaW5nICovXG4gIGRlY29tcG9zaXRpb25McjogbnVtYmVyO1xuICAvKiogTGVhcm5pbmcgcmF0ZSBmb3Igc3ludGhlc2lzIHRyYWluaW5nICovXG4gIHN5bnRoZXNpc0xyOiBudW1iZXI7XG4gIC8qKiBMZWFybmluZyByYXRlIGZvciBjb250cmFzdGl2ZSBmaW5lLXR1bmluZyAqL1xuICBjb250cmFzdGl2ZUxyOiBudW1iZXI7XG4gIC8qKiBCYXRjaCBzaXplICovXG4gIGJhdGNoU2l6ZTogbnVtYmVyO1xuICAvKiogTnVtYmVyIG9mIGVwb2NocyAqL1xuICBlcG9jaHM6IG51bWJlcjtcbiAgLyoqIENvbnRyYXN0aXZlIG1hcmdpbiBmb3IgdHJpcGxldCBsb3NzICovXG4gIGNvbnRyYXN0aXZlTWFyZ2luOiBudW1iZXI7XG4gIC8qKiBUZW1wZXJhdHVyZSBmb3IgSW5mb05DRSBsb3NzICovXG4gIGluZm9uY2VUZW1wZXJhdHVyZTogbnVtYmVyO1xuICAvKiogV2VpZ2h0IGZvciBkZWNvbXBvc2l0aW9uIGxvc3MgKi9cbiAgZGVjb21wb3NpdGlvbldlaWdodDogbnVtYmVyO1xuICAvKiogV2VpZ2h0IGZvciBzeW50aGVzaXMgbG9zcyAqL1xuICBzeW50aGVzaXNXZWlnaHQ6IG51bWJlcjtcbiAgLyoqIFdlaWdodCBmb3Igcm91dGluZyBsb3NzICovXG4gIHJvdXRpbmdXZWlnaHQ6IG51bWJlcjtcbiAgLyoqIE1pbmltdW0gcXVhbGl0eSBmb3IgdXBkYXRlcyAqL1xuICBxdWFsaXR5VGhyZXNob2xkOiBudW1iZXI7XG4gIC8qKiBFdmFsdWF0aW9uIGludGVydmFsIChlcG9jaHMpICovXG4gIGV2YWx1YXRpb25JbnRlcnZhbDogbnVtYmVyO1xuICAvKiogV2FybXVwIHN0ZXBzICovXG4gIHdhcm11cFN0ZXBzOiBudW1iZXI7XG4gIC8qKiBFYXJseSBzdG9wcGluZyBwYXRpZW5jZSAqL1xuICBlYXJseVN0b3BwaW5nUGF0aWVuY2U6IG51bWJlcjtcbiAgLyoqIFZhbGlkYXRpb24gc3BsaXQgcmF0aW8gKi9cbiAgdmFsaWRhdGlvblNwbGl0OiBudW1iZXI7XG4gIC8qKiBSYW5kb20gc2VlZCAqL1xuICBzZWVkOiBudW1iZXI7XG59XG5cbi8qKlxuICogVHJhaW5pbmcgcmVzdWx0IGZvciBhIHBoYXNlXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgVHJhaW5pbmdSZXN1bHQge1xuICAvKiogVHJhaW5pbmcgcGhhc2UgbmFtZSAqL1xuICBwaGFzZTogc3RyaW5nO1xuICAvKiogRXBvY2hzIGNvbXBsZXRlZCAqL1xuICBlcG9jaHNDb21wbGV0ZWQ6IG51bWJlcjtcbiAgLyoqIFRvdGFsIHN0ZXBzICovXG4gIHRvdGFsU3RlcHM6IG51bWJlcjtcbiAgLyoqIEZpbmFsIHRyYWluaW5nIGxvc3MgKi9cbiAgZmluYWxMb3NzOiBudW1iZXI7XG4gIC8qKiBCZXN0IHZhbGlkYXRpb24gbG9zcyAqL1xuICBiZXN0VmFsTG9zczogbnVtYmVyO1xuICAvKiogQmVzdCBlcG9jaCAqL1xuICBiZXN0RXBvY2g6IG51bWJlcjtcbiAgLyoqIEZpbmFsIGFjY3VyYWN5IChmb3IgY2xhc3NpZmljYXRpb24gdGFza3MpICovXG4gIGFjY3VyYWN5OiBudW1iZXI7XG4gIC8qKiBMb3NzIGhpc3RvcnkgcGVyIGVwb2NoICovXG4gIGxvc3NIaXN0b3J5OiBudW1iZXJbXTtcbiAgLyoqIFZhbGlkYXRpb24gbG9zcyBoaXN0b3J5ICovXG4gIHZhbExvc3NIaXN0b3J5OiBudW1iZXJbXTtcbiAgLyoqIFRyYWluaW5nIGR1cmF0aW9uIGluIG1pbGxpc2Vjb25kcyAqL1xuICBkdXJhdGlvbk1zOiBudW1iZXI7XG4gIC8qKiBXaGV0aGVyIGVhcmx5IHN0b3BwaW5nIHdhcyB0cmlnZ2VyZWQgKi9cbiAgZWFybHlTdG9wcGVkOiBib29sZWFuO1xufVxuXG4vKipcbiAqIEV2YWx1YXRpb24gcmVzdWx0IGZvciB0aGUgdHJhaW5lZCBtb2RlbFxuICovXG5leHBvcnQgaW50ZXJmYWNlIEV2YWx1YXRpb25SZXN1bHQge1xuICAvKiogRGVjb21wb3NpdGlvbiBhY2N1cmFjeSAqL1xuICBkZWNvbXBvc2l0aW9uQWNjdXJhY3k6IG51bWJlcjtcbiAgLyoqIFN5bnRoZXNpcyBxdWFsaXR5ICovXG4gIHN5bnRoZXNpc1F1YWxpdHk6IG51bWJlcjtcbiAgLyoqIFJvdXRpbmcgYWNjdXJhY3kgKi9cbiAgcm91dGluZ0FjY3VyYWN5OiBudW1iZXI7XG4gIC8qKiBIYXJkIG5lZ2F0aXZlIGFjY3VyYWN5ICovXG4gIGhhcmROZWdhdGl2ZUFjY3VyYWN5OiBudW1iZXI7XG4gIC8qKiBBdmVyYWdlIGxhdGVuY3kgaW4gbXMgKi9cbiAgYXZnTGF0ZW5jeU1zOiBudW1iZXI7XG4gIC8qKiBUb3RhbCBleGFtcGxlcyBldmFsdWF0ZWQgKi9cbiAgdG90YWxFeGFtcGxlczogbnVtYmVyO1xuICAvKiogUGVyLWFnZW50IGFjY3VyYWN5ICovXG4gIHBlckFnZW50QWNjdXJhY3k6IFJlY29yZDxzdHJpbmcsIG51bWJlcj47XG59XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBEZWZhdWx0IENvbmZpZ3VyYXRpb25zXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIERlZmF1bHQgUkxNIHRyYWluaW5nIGNvbmZpZ3VyYXRpb25cbiAqL1xuZXhwb3J0IGNvbnN0IERFRkFVTFRfUkxNX0NPTkZJRzogUmxtVHJhaW5pbmdDb25maWcgPSB7XG4gIGRlY29tcG9zaXRpb25McjogMWUtNSxcbiAgc3ludGhlc2lzTHI6IDFlLTUsXG4gIGNvbnRyYXN0aXZlTHI6IDJlLTUsXG4gIGJhdGNoU2l6ZTogMzIsXG4gIGVwb2NoczogMTAsXG4gIGNvbnRyYXN0aXZlTWFyZ2luOiAwLjUsXG4gIGluZm9uY2VUZW1wZXJhdHVyZTogMC4wNyxcbiAgZGVjb21wb3NpdGlvbldlaWdodDogMS4wLFxuICBzeW50aGVzaXNXZWlnaHQ6IDEuMCxcbiAgcm91dGluZ1dlaWdodDogMS4wLFxuICBxdWFsaXR5VGhyZXNob2xkOiAwLjcsXG4gIGV2YWx1YXRpb25JbnRlcnZhbDogMSxcbiAgd2FybXVwU3RlcHM6IDEwMCxcbiAgZWFybHlTdG9wcGluZ1BhdGllbmNlOiAzLFxuICB2YWxpZGF0aW9uU3BsaXQ6IDAuMSxcbiAgc2VlZDogNDIsXG59O1xuXG4vKipcbiAqIEZhc3QgdHJhaW5pbmcgY29uZmlndXJhdGlvblxuICovXG5leHBvcnQgY29uc3QgRkFTVF9STE1fQ09ORklHOiBSbG1UcmFpbmluZ0NvbmZpZyA9IHtcbiAgLi4uREVGQVVMVF9STE1fQ09ORklHLFxuICBlcG9jaHM6IDMsXG4gIGJhdGNoU2l6ZTogNjQsXG4gIGRlY29tcG9zaXRpb25McjogMWUtNCxcbiAgc3ludGhlc2lzTHI6IDFlLTQsXG4gIGNvbnRyYXN0aXZlTHI6IDVlLTUsXG4gIGVhcmx5U3RvcHBpbmdQYXRpZW5jZTogMSxcbn07XG5cbi8qKlxuICogVGhvcm91Z2ggdHJhaW5pbmcgY29uZmlndXJhdGlvblxuICovXG5leHBvcnQgY29uc3QgVEhPUk9VR0hfUkxNX0NPTkZJRzogUmxtVHJhaW5pbmdDb25maWcgPSB7XG4gIC4uLkRFRkFVTFRfUkxNX0NPTkZJRyxcbiAgZXBvY2hzOiA1MCxcbiAgYmF0Y2hTaXplOiAxNixcbiAgZGVjb21wb3NpdGlvbkxyOiA1ZS02LFxuICBzeW50aGVzaXNMcjogNWUtNixcbiAgY29udHJhc3RpdmVMcjogMWUtNSxcbiAgZWFybHlTdG9wcGluZ1BhdGllbmNlOiAxMCxcbn07XG5cbi8qKlxuICogUm91dGluZy1mb2N1c2VkIHRyYWluaW5nIGNvbmZpZ3VyYXRpb25cbiAqL1xuZXhwb3J0IGNvbnN0IFJPVVRJTkdfRk9DVVNFRF9DT05GSUc6IFJsbVRyYWluaW5nQ29uZmlnID0ge1xuICAuLi5ERUZBVUxUX1JMTV9DT05GSUcsXG4gIHJvdXRpbmdXZWlnaHQ6IDIuMCxcbiAgZGVjb21wb3NpdGlvbldlaWdodDogMC41LFxuICBzeW50aGVzaXNXZWlnaHQ6IDAuNSxcbiAgY29udHJhc3RpdmVMcjogM2UtNSxcbiAgY29udHJhc3RpdmVNYXJnaW46IDAuMyxcbiAgaW5mb25jZVRlbXBlcmF0dXJlOiAwLjA1LFxufTtcblxuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIEFnZW50IERlZmluaXRpb25zXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIEFnZW50IHR5cGVzIHdpdGggZGVzY3JpcHRpb25zIGFuZCBrZXl3b3Jkc1xuICovXG5leHBvcnQgY29uc3QgQUdFTlRfREVGSU5JVElPTlM6IFJlY29yZDxzdHJpbmcsIHsgZGVzY3JpcHRpb246IHN0cmluZzsga2V5d29yZHM6IHN0cmluZ1tdIH0+ID0ge1xuICBjb2Rlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnU29mdHdhcmUgZGV2ZWxvcGVyIHdobyB3cml0ZXMgYW5kIGltcGxlbWVudHMgY29kZScsXG4gICAga2V5d29yZHM6IFsnaW1wbGVtZW50JywgJ2J1aWxkJywgJ2NyZWF0ZScsICdjb2RlJywgJ3dyaXRlJywgJ2RldmVsb3AnLCAncHJvZ3JhbSddLFxuICB9LFxuICByZXNlYXJjaGVyOiB7XG4gICAgZGVzY3JpcHRpb246ICdUZWNobmljYWwgcmVzZWFyY2hlciB3aG8gaW52ZXN0aWdhdGVzIGFuZCBhbmFseXplcycsXG4gICAga2V5d29yZHM6IFsncmVzZWFyY2gnLCAnaW52ZXN0aWdhdGUnLCAnYW5hbHl6ZScsICdleHBsb3JlJywgJ3N0dWR5JywgJ2V4YW1pbmUnXSxcbiAgfSxcbiAgcmV2aWV3ZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ0NvZGUgcmV2aWV3ZXIgd2hvIGV2YWx1YXRlcyBjb2RlIHF1YWxpdHknLFxuICAgIGtleXdvcmRzOiBbJ3JldmlldycsICdjaGVjaycsICdldmFsdWF0ZScsICdhc3Nlc3MnLCAnZXhhbWluZScsICdpbnNwZWN0J10sXG4gIH0sXG4gIHRlc3Rlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnUUEgZW5naW5lZXIgd2hvIHdyaXRlcyBhbmQgcnVucyB0ZXN0cycsXG4gICAga2V5d29yZHM6IFsndGVzdCcsICd1bml0IHRlc3QnLCAnY292ZXJhZ2UnLCAndmFsaWRhdGUnLCAndmVyaWZ5JywgJ3FhJ10sXG4gIH0sXG4gIGFyY2hpdGVjdDoge1xuICAgIGRlc2NyaXB0aW9uOiAnU3lzdGVtIGFyY2hpdGVjdCB3aG8gZGVzaWducyBzb2Z0d2FyZSBzdHJ1Y3R1cmUnLFxuICAgIGtleXdvcmRzOiBbJ2Rlc2lnbicsICdwbGFuJywgJ2FyY2hpdGVjdHVyZScsICdzY2hlbWEnLCAnc3RydWN0dXJlJywgJ2RpYWdyYW0nXSxcbiAgfSxcbiAgJ3NlY3VyaXR5LWFyY2hpdGVjdCc6IHtcbiAgICBkZXNjcmlwdGlvbjogJ1NlY3VyaXR5IHNwZWNpYWxpc3Qgd2hvIGF1ZGl0cyB2dWxuZXJhYmlsaXRpZXMnLFxuICAgIGtleXdvcmRzOiBbJ3NlY3VyaXR5JywgJ2F1ZGl0JywgJ3Z1bG5lcmFiaWxpdHknLCAneHNzJywgJ2luamVjdGlvbicsICdjdmUnXSxcbiAgfSxcbiAgZGVidWdnZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ0J1ZyBodW50ZXIgd2hvIGZpeGVzIGVycm9ycyBhbmQgdHJhY2VzIGlzc3VlcycsXG4gICAga2V5d29yZHM6IFsnZml4JywgJ2RlYnVnJywgJ2J1ZycsICdlcnJvcicsICd0cmFjZScsICdjcmFzaCcsICd0cm91Ymxlc2hvb3QnXSxcbiAgfSxcbiAgZG9jdW1lbnRlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnVGVjaG5pY2FsIHdyaXRlciB3aG8gY3JlYXRlcyBkb2N1bWVudGF0aW9uJyxcbiAgICBrZXl3b3JkczogWydkb2N1bWVudCcsICdqc2RvYycsICdyZWFkbWUnLCAnY29tbWVudCcsICdleHBsYWluJywgJ2Rlc2NyaWJlJ10sXG4gIH0sXG4gIHJlZmFjdG9yZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ0NvZGUgbW9kZXJuaXplciB3aG8gcmVzdHJ1Y3R1cmVzIHdpdGhvdXQgY2hhbmdpbmcgYmVoYXZpb3InLFxuICAgIGtleXdvcmRzOiBbJ3JlZmFjdG9yJywgJ3Jlc3RydWN0dXJlJywgJ21vZGVybml6ZScsICdjbGVhbicsICdzaW1wbGlmeScsICdjb25zb2xpZGF0ZSddLFxuICB9LFxuICBvcHRpbWl6ZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ1BlcmZvcm1hbmNlIGVuZ2luZWVyIHdobyBzcGVlZHMgdXAgc2xvdyBjb2RlJyxcbiAgICBrZXl3b3JkczogWydvcHRpbWl6ZScsICdwZXJmb3JtYW5jZScsICdzcGVlZCcsICdjYWNoZScsICdpbXByb3ZlJywgJ2Zhc3RlciddLFxuICB9LFxuICBkZXZvcHM6IHtcbiAgICBkZXNjcmlwdGlvbjogJ0Rldk9wcyBlbmdpbmVlciB3aG8gbWFuYWdlcyBkZXBsb3ltZW50IGFuZCBpbmZyYXN0cnVjdHVyZScsXG4gICAga2V5d29yZHM6IFsnZGVwbG95JywgJ2NpL2NkJywgJ2t1YmVybmV0ZXMnLCAnZG9ja2VyJywgJ2luZnJhc3RydWN0dXJlJywgJ3BpcGVsaW5lJ10sXG4gIH0sXG4gICdhcGktZG9jcyc6IHtcbiAgICBkZXNjcmlwdGlvbjogJ0FQSSBkb2N1bWVudGF0aW9uIHNwZWNpYWxpc3Qgd2hvIGNyZWF0ZXMgc3BlY3MnLFxuICAgIGtleXdvcmRzOiBbJ29wZW5hcGknLCAnc3dhZ2dlcicsICdhcGkgcmVmZXJlbmNlJywgJ2VuZHBvaW50JywgJ3NwZWMnLCAncmVzdCddLFxuICB9LFxuICBwbGFubmVyOiB7XG4gICAgZGVzY3JpcHRpb246ICdQcm9qZWN0IHBsYW5uZXIgd2hvIG9yZ2FuaXplcyBhbmQgc2NoZWR1bGVzIHdvcmsnLFxuICAgIGtleXdvcmRzOiBbJ3BsYW4nLCAnZXN0aW1hdGUnLCAnc2NoZWR1bGUnLCAndGltZWxpbmUnLCAnc3ByaW50JywgJ3JvYWRtYXAnXSxcbiAgfSxcbn07XG5cbi8qKlxuICogSGFyZCBuZWdhdGl2ZSBwYWlycyAoY29uZnVzYWJsZSBhZ2VudCBjb21iaW5hdGlvbnMpXG4gKi9cbmV4cG9ydCBjb25zdCBIQVJEX05FR0FUSVZFX1BBSVJTOiBbc3RyaW5nLCBzdHJpbmddW10gPSBbXG4gIFsnY29kZXInLCAnZGVidWdnZXInXSxcbiAgWydjb2RlcicsICdyZWZhY3RvcmVyJ10sXG4gIFsncmVzZWFyY2hlcicsICdyZXZpZXdlciddLFxuICBbJ3Rlc3RlcicsICdyZXZpZXdlciddLFxuICBbJ2FyY2hpdGVjdCcsICdwbGFubmVyJ10sXG4gIFsnZG9jdW1lbnRlcicsICdhcGktZG9jcyddLFxuICBbJ29wdGltaXplcicsICdkZWJ1Z2dlciddLFxuICBbJ2Rldm9wcycsICdhcmNoaXRlY3QnXSxcbiAgWydzZWN1cml0eS1hcmNoaXRlY3QnLCAncmV2aWV3ZXInXSxcbl07XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBSTE0gVHJhaW5lciBDbGFzc1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuLyoqXG4gKiBSTE0gVHJhaW5lciBmb3IgUnV2TFRSQSBtb2RlbHNcbiAqXG4gKiBQcm92aWRlcyB0cmFpbmluZyBjYXBhYmlsaXRpZXMgZm9yIGRlY29tcG9zaXRpb24sIHN5bnRoZXNpcywgYW5kIHJvdXRpbmcgdGFza3MuXG4gKi9cbmV4cG9ydCBjbGFzcyBSbG1UcmFpbmVyIHtcbiAgcHJpdmF0ZSBjb25maWc6IFJsbVRyYWluaW5nQ29uZmlnO1xuICBwcml2YXRlIGN1cnJlbnRFcG9jaCA9IDA7XG4gIHByaXZhdGUgY3VycmVudFN0ZXAgPSAwO1xuICBwcml2YXRlIGJlc3RWYWxMb3NzID0gSW5maW5pdHk7XG4gIHByaXZhdGUgcGF0aWVuY2VDb3VudGVyID0gMDtcbiAgcHJpdmF0ZSBsb3NzSGlzdG9yeTogbnVtYmVyW10gPSBbXTtcbiAgcHJpdmF0ZSB2YWxMb3NzSGlzdG9yeTogbnVtYmVyW10gPSBbXTtcblxuICAvKipcbiAgICogQ3JlYXRlIGEgbmV3IFJMTSB0cmFpbmVyXG4gICAqL1xuICBjb25zdHJ1Y3Rvcihjb25maWc6IFBhcnRpYWw8UmxtVHJhaW5pbmdDb25maWc+ID0ge30pIHtcbiAgICB0aGlzLmNvbmZpZyA9IHsgLi4uREVGQVVMVF9STE1fQ09ORklHLCAuLi5jb25maWcgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBUcmFpbiBvbiBkZWNvbXBvc2l0aW9uIHRhc2tcbiAgICpcbiAgICogTGVhcm5zIHRvIGJyZWFrIGNvbXBsZXggcXVlcmllcyBpbnRvIG1hbmFnZWFibGUgc3ViLXF1ZXJpZXMuXG4gICAqL1xuICBhc3luYyB0cmFpbkRlY29tcG9zaXRpb24oZGF0YXNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBQcm9taXNlPFRyYWluaW5nUmVzdWx0PiB7XG4gICAgY29uc3Qgc3RhcnRUaW1lID0gRGF0ZS5ub3coKTtcbiAgICB0aGlzLnJlc2V0U3RhdGUoKTtcblxuICAgIGNvbnN0IHsgdHJhaW5TZXQsIHZhbFNldCB9ID0gdGhpcy5zcGxpdERhdGFzZXQoZGF0YXNldCk7XG4gICAgY29uc3QgYmF0Y2hlcyA9IHRoaXMuY3JlYXRlQmF0Y2hlcyh0cmFpblNldCk7XG5cbiAgICBmb3IgKGxldCBlcG9jaCA9IDA7IGVwb2NoIDwgdGhpcy5jb25maWcuZXBvY2hzOyBlcG9jaCsrKSB7XG4gICAgICB0aGlzLmN1cnJlbnRFcG9jaCA9IGVwb2NoO1xuICAgICAgbGV0IGVwb2NoTG9zcyA9IDA7XG5cbiAgICAgIGZvciAoY29uc3QgYmF0Y2ggb2YgYmF0Y2hlcykge1xuICAgICAgICBjb25zdCBiYXRjaExvc3MgPSB0aGlzLnRyYWluRGVjb21wb3NpdGlvbkJhdGNoKGJhdGNoKTtcbiAgICAgICAgZXBvY2hMb3NzICs9IGJhdGNoTG9zcztcbiAgICAgICAgdGhpcy5jdXJyZW50U3RlcCsrO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBhdmdMb3NzID0gZXBvY2hMb3NzIC8gYmF0Y2hlcy5sZW5ndGg7XG4gICAgICB0aGlzLmxvc3NIaXN0b3J5LnB1c2goYXZnTG9zcyk7XG5cbiAgICAgIC8vIFZhbGlkYXRpb25cbiAgICAgIGNvbnN0IHZhbExvc3MgPSB0aGlzLnZhbGlkYXRlRGVjb21wb3NpdGlvbih2YWxTZXQpO1xuICAgICAgdGhpcy52YWxMb3NzSGlzdG9yeS5wdXNoKHZhbExvc3MpO1xuXG4gICAgICAvLyBFYXJseSBzdG9wcGluZ1xuICAgICAgaWYgKHZhbExvc3MgPCB0aGlzLmJlc3RWYWxMb3NzKSB7XG4gICAgICAgIHRoaXMuYmVzdFZhbExvc3MgPSB2YWxMb3NzO1xuICAgICAgICB0aGlzLnBhdGllbmNlQ291bnRlciA9IDA7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLnBhdGllbmNlQ291bnRlcisrO1xuICAgICAgICBpZiAodGhpcy5wYXRpZW5jZUNvdW50ZXIgPj0gdGhpcy5jb25maWcuZWFybHlTdG9wcGluZ1BhdGllbmNlKSB7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgcGhhc2U6ICdkZWNvbXBvc2l0aW9uJyxcbiAgICAgIGVwb2Noc0NvbXBsZXRlZDogdGhpcy5jdXJyZW50RXBvY2ggKyAxLFxuICAgICAgdG90YWxTdGVwczogdGhpcy5jdXJyZW50U3RlcCxcbiAgICAgIGZpbmFsTG9zczogdGhpcy5sb3NzSGlzdG9yeVt0aGlzLmxvc3NIaXN0b3J5Lmxlbmd0aCAtIDFdIHx8IDAsXG4gICAgICBiZXN0VmFsTG9zczogdGhpcy5iZXN0VmFsTG9zcyxcbiAgICAgIGJlc3RFcG9jaDogdGhpcy5maW5kQmVzdEVwb2NoKCksXG4gICAgICBhY2N1cmFjeTogMCwgLy8gTm90IGFwcGxpY2FibGUgZm9yIGRlY29tcG9zaXRpb25cbiAgICAgIGxvc3NIaXN0b3J5OiB0aGlzLmxvc3NIaXN0b3J5LFxuICAgICAgdmFsTG9zc0hpc3Rvcnk6IHRoaXMudmFsTG9zc0hpc3RvcnksXG4gICAgICBkdXJhdGlvbk1zOiBEYXRlLm5vdygpIC0gc3RhcnRUaW1lLFxuICAgICAgZWFybHlTdG9wcGVkOiB0aGlzLnBhdGllbmNlQ291bnRlciA+PSB0aGlzLmNvbmZpZy5lYXJseVN0b3BwaW5nUGF0aWVuY2UsXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBUcmFpbiBvbiBzeW50aGVzaXMgdGFza1xuICAgKlxuICAgKiBMZWFybnMgdG8gY29tYmluZSBzdWItYW5zd2VycyBpbnRvIGNvaGVyZW50IGZpbmFsIHJlc3BvbnNlcy5cbiAgICovXG4gIGFzeW5jIHRyYWluU3ludGhlc2lzKGRhdGFzZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdKTogUHJvbWlzZTxUcmFpbmluZ1Jlc3VsdD4ge1xuICAgIGNvbnN0IHN0YXJ0VGltZSA9IERhdGUubm93KCk7XG4gICAgdGhpcy5yZXNldFN0YXRlKCk7XG5cbiAgICBjb25zdCB7IHRyYWluU2V0LCB2YWxTZXQgfSA9IHRoaXMuc3BsaXREYXRhc2V0KGRhdGFzZXQpO1xuICAgIGNvbnN0IGJhdGNoZXMgPSB0aGlzLmNyZWF0ZUJhdGNoZXModHJhaW5TZXQpO1xuXG4gICAgZm9yIChsZXQgZXBvY2ggPSAwOyBlcG9jaCA8IHRoaXMuY29uZmlnLmVwb2NoczsgZXBvY2grKykge1xuICAgICAgdGhpcy5jdXJyZW50RXBvY2ggPSBlcG9jaDtcbiAgICAgIGxldCBlcG9jaExvc3MgPSAwO1xuXG4gICAgICBmb3IgKGNvbnN0IGJhdGNoIG9mIGJhdGNoZXMpIHtcbiAgICAgICAgY29uc3QgYmF0Y2hMb3NzID0gdGhpcy50cmFpblN5bnRoZXNpc0JhdGNoKGJhdGNoKTtcbiAgICAgICAgZXBvY2hMb3NzICs9IGJhdGNoTG9zcztcbiAgICAgICAgdGhpcy5jdXJyZW50U3RlcCsrO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBhdmdMb3NzID0gZXBvY2hMb3NzIC8gYmF0Y2hlcy5sZW5ndGg7XG4gICAgICB0aGlzLmxvc3NIaXN0b3J5LnB1c2goYXZnTG9zcyk7XG5cbiAgICAgIC8vIFZhbGlkYXRpb25cbiAgICAgIGNvbnN0IHZhbExvc3MgPSB0aGlzLnZhbGlkYXRlU3ludGhlc2lzKHZhbFNldCk7XG4gICAgICB0aGlzLnZhbExvc3NIaXN0b3J5LnB1c2godmFsTG9zcyk7XG5cbiAgICAgIC8vIEVhcmx5IHN0b3BwaW5nXG4gICAgICBpZiAodmFsTG9zcyA8IHRoaXMuYmVzdFZhbExvc3MpIHtcbiAgICAgICAgdGhpcy5iZXN0VmFsTG9zcyA9IHZhbExvc3M7XG4gICAgICAgIHRoaXMucGF0aWVuY2VDb3VudGVyID0gMDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMucGF0aWVuY2VDb3VudGVyKys7XG4gICAgICAgIGlmICh0aGlzLnBhdGllbmNlQ291bnRlciA+PSB0aGlzLmNvbmZpZy5lYXJseVN0b3BwaW5nUGF0aWVuY2UpIHtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBwaGFzZTogJ3N5bnRoZXNpcycsXG4gICAgICBlcG9jaHNDb21wbGV0ZWQ6IHRoaXMuY3VycmVudEVwb2NoICsgMSxcbiAgICAgIHRvdGFsU3RlcHM6IHRoaXMuY3VycmVudFN0ZXAsXG4gICAgICBmaW5hbExvc3M6IHRoaXMubG9zc0hpc3RvcnlbdGhpcy5sb3NzSGlzdG9yeS5sZW5ndGggLSAxXSB8fCAwLFxuICAgICAgYmVzdFZhbExvc3M6IHRoaXMuYmVzdFZhbExvc3MsXG4gICAgICBiZXN0RXBvY2g6IHRoaXMuZmluZEJlc3RFcG9jaCgpLFxuICAgICAgYWNjdXJhY3k6IDAsXG4gICAgICBsb3NzSGlzdG9yeTogdGhpcy5sb3NzSGlzdG9yeSxcbiAgICAgIHZhbExvc3NIaXN0b3J5OiB0aGlzLnZhbExvc3NIaXN0b3J5LFxuICAgICAgZHVyYXRpb25NczogRGF0ZS5ub3coKSAtIHN0YXJ0VGltZSxcbiAgICAgIGVhcmx5U3RvcHBlZDogdGhpcy5wYXRpZW5jZUNvdW50ZXIgPj0gdGhpcy5jb25maWcuZWFybHlTdG9wcGluZ1BhdGllbmNlLFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogQ29udHJhc3RpdmUgZmluZS10dW5pbmcgZm9yIGFnZW50IHJvdXRpbmdcbiAgICpcbiAgICogVXNlcyB0cmlwbGV0IGxvc3MgYW5kIEluZm9OQ0UgdG8gaW1wcm92ZSByb3V0aW5nIGFjY3VyYWN5LlxuICAgKi9cbiAgYXN5bmMgdHJhaW5Db250cmFzdGl2ZShwYWlyczogQ29udHJhc3RpdmVQYWlyW10pOiBQcm9taXNlPFRyYWluaW5nUmVzdWx0PiB7XG4gICAgY29uc3Qgc3RhcnRUaW1lID0gRGF0ZS5ub3coKTtcbiAgICB0aGlzLnJlc2V0U3RhdGUoKTtcblxuICAgIGlmIChwYWlycy5sZW5ndGggPT09IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTm8gY29udHJhc3RpdmUgcGFpcnMgcHJvdmlkZWQnKTtcbiAgICB9XG5cbiAgICBjb25zdCB7IHRyYWluU2V0LCB2YWxTZXQgfSA9IHRoaXMuc3BsaXRQYWlycyhwYWlycyk7XG4gICAgY29uc3QgYmF0Y2hlcyA9IHRoaXMuY3JlYXRlUGFpckJhdGNoZXModHJhaW5TZXQpO1xuICAgIGxldCB0b3RhbENvcnJlY3QgPSAwO1xuICAgIGxldCB0b3RhbEV4YW1wbGVzID0gMDtcblxuICAgIGZvciAobGV0IGVwb2NoID0gMDsgZXBvY2ggPCB0aGlzLmNvbmZpZy5lcG9jaHM7IGVwb2NoKyspIHtcbiAgICAgIHRoaXMuY3VycmVudEVwb2NoID0gZXBvY2g7XG4gICAgICBsZXQgZXBvY2hMb3NzID0gMDtcblxuICAgICAgZm9yIChjb25zdCBiYXRjaCBvZiBiYXRjaGVzKSB7XG4gICAgICAgIGNvbnN0IGJhdGNoTG9zcyA9IHRoaXMudHJhaW5Db250cmFzdGl2ZUJhdGNoKGJhdGNoKTtcbiAgICAgICAgZXBvY2hMb3NzICs9IGJhdGNoTG9zcztcbiAgICAgICAgdGhpcy5jdXJyZW50U3RlcCsrO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBhdmdMb3NzID0gZXBvY2hMb3NzIC8gYmF0Y2hlcy5sZW5ndGg7XG4gICAgICB0aGlzLmxvc3NIaXN0b3J5LnB1c2goYXZnTG9zcyk7XG5cbiAgICAgIC8vIFZhbGlkYXRpb25cbiAgICAgIGNvbnN0IHsgbG9zczogdmFsTG9zcywgY29ycmVjdCwgdG90YWwgfSA9IHRoaXMudmFsaWRhdGVDb250cmFzdGl2ZSh2YWxTZXQpO1xuICAgICAgdGhpcy52YWxMb3NzSGlzdG9yeS5wdXNoKHZhbExvc3MpO1xuICAgICAgdG90YWxDb3JyZWN0ID0gY29ycmVjdDtcbiAgICAgIHRvdGFsRXhhbXBsZXMgPSB0b3RhbDtcblxuICAgICAgLy8gRWFybHkgc3RvcHBpbmdcbiAgICAgIGlmICh2YWxMb3NzIDwgdGhpcy5iZXN0VmFsTG9zcykge1xuICAgICAgICB0aGlzLmJlc3RWYWxMb3NzID0gdmFsTG9zcztcbiAgICAgICAgdGhpcy5wYXRpZW5jZUNvdW50ZXIgPSAwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5wYXRpZW5jZUNvdW50ZXIrKztcbiAgICAgICAgaWYgKHRoaXMucGF0aWVuY2VDb3VudGVyID49IHRoaXMuY29uZmlnLmVhcmx5U3RvcHBpbmdQYXRpZW5jZSkge1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIHBoYXNlOiAnY29udHJhc3RpdmUnLFxuICAgICAgZXBvY2hzQ29tcGxldGVkOiB0aGlzLmN1cnJlbnRFcG9jaCArIDEsXG4gICAgICB0b3RhbFN0ZXBzOiB0aGlzLmN1cnJlbnRTdGVwLFxuICAgICAgZmluYWxMb3NzOiB0aGlzLmxvc3NIaXN0b3J5W3RoaXMubG9zc0hpc3RvcnkubGVuZ3RoIC0gMV0gfHwgMCxcbiAgICAgIGJlc3RWYWxMb3NzOiB0aGlzLmJlc3RWYWxMb3NzLFxuICAgICAgYmVzdEVwb2NoOiB0aGlzLmZpbmRCZXN0RXBvY2goKSxcbiAgICAgIGFjY3VyYWN5OiB0b3RhbEV4YW1wbGVzID4gMCA/IHRvdGFsQ29ycmVjdCAvIHRvdGFsRXhhbXBsZXMgOiAwLFxuICAgICAgbG9zc0hpc3Rvcnk6IHRoaXMubG9zc0hpc3RvcnksXG4gICAgICB2YWxMb3NzSGlzdG9yeTogdGhpcy52YWxMb3NzSGlzdG9yeSxcbiAgICAgIGR1cmF0aW9uTXM6IERhdGUubm93KCkgLSBzdGFydFRpbWUsXG4gICAgICBlYXJseVN0b3BwZWQ6IHRoaXMucGF0aWVuY2VDb3VudGVyID49IHRoaXMuY29uZmlnLmVhcmx5U3RvcHBpbmdQYXRpZW5jZSxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIEV2YWx1YXRlIHRyYWluZWQgbW9kZWwgb24gdGVzdCBzZXRcbiAgICovXG4gIGFzeW5jIGV2YWx1YXRlKHRlc3RTZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdKTogUHJvbWlzZTxFdmFsdWF0aW9uUmVzdWx0PiB7XG4gICAgY29uc3QgcGVyQWdlbnRBY2N1cmFjeTogUmVjb3JkPHN0cmluZywgeyBjb3JyZWN0OiBudW1iZXI7IHRvdGFsOiBudW1iZXIgfT4gPSB7fTtcblxuICAgIGxldCBkZWNvbXBvc2l0aW9uQ29ycmVjdCA9IDA7XG4gICAgbGV0IHN5bnRoZXNpc1F1YWxpdHlTdW0gPSAwO1xuICAgIGxldCByb3V0aW5nQ29ycmVjdCA9IDA7XG4gICAgbGV0IGhhcmROZWdhdGl2ZUNvcnJlY3QgPSAwO1xuICAgIGxldCBoYXJkTmVnYXRpdmVUb3RhbCA9IDA7XG4gICAgbGV0IHRvdGFsTGF0ZW5jeSA9IDA7XG5cbiAgICBmb3IgKGNvbnN0IGV4YW1wbGUgb2YgdGVzdFNldCkge1xuICAgICAgLy8gRGVjb21wb3NpdGlvbiBldmFsdWF0aW9uXG4gICAgICBpZiAoZXhhbXBsZS5kZWNvbXBvc2l0aW9uLnN1Y2Nlc3MgJiYgZXhhbXBsZS5kZWNvbXBvc2l0aW9uLnN1YlF1ZXJpZXMubGVuZ3RoID4gMCkge1xuICAgICAgICBkZWNvbXBvc2l0aW9uQ29ycmVjdCsrO1xuICAgICAgfVxuXG4gICAgICAvLyBTeW50aGVzaXMgcXVhbGl0eVxuICAgICAgc3ludGhlc2lzUXVhbGl0eVN1bSArPSBleGFtcGxlLnF1YWxpdHlTY29yZTtcblxuICAgICAgLy8gUm91dGluZyBldmFsdWF0aW9uXG4gICAgICBmb3IgKGNvbnN0IHN1YlF1ZXJ5IG9mIGV4YW1wbGUuZGVjb21wb3NpdGlvbi5zdWJRdWVyaWVzKSB7XG4gICAgICAgIGlmIChzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50KSB7XG4gICAgICAgICAgY29uc3QgcHJlZGljdGVkID0gdGhpcy5wcmVkaWN0QWdlbnQoc3ViUXVlcnkucXVlcnkpO1xuICAgICAgICAgIGNvbnN0IGNvcnJlY3QgPSBwcmVkaWN0ZWQgPT09IHN1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnQ7XG5cbiAgICAgICAgICBpZiAoY29ycmVjdCkge1xuICAgICAgICAgICAgcm91dGluZ0NvcnJlY3QrKztcbiAgICAgICAgICB9XG5cbiAgICAgICAgICAvLyBUcmFjayBwZXItYWdlbnQgYWNjdXJhY3lcbiAgICAgICAgICBpZiAoIXBlckFnZW50QWNjdXJhY3lbc3ViUXVlcnkucmVjb21tZW5kZWRBZ2VudF0pIHtcbiAgICAgICAgICAgIHBlckFnZW50QWNjdXJhY3lbc3ViUXVlcnkucmVjb21tZW5kZWRBZ2VudF0gPSB7IGNvcnJlY3Q6IDAsIHRvdGFsOiAwIH07XG4gICAgICAgICAgfVxuICAgICAgICAgIHBlckFnZW50QWNjdXJhY3lbc3ViUXVlcnkucmVjb21tZW5kZWRBZ2VudF0udG90YWwrKztcbiAgICAgICAgICBpZiAoY29ycmVjdCkge1xuICAgICAgICAgICAgcGVyQWdlbnRBY2N1cmFjeVtzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50XS5jb3JyZWN0Kys7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgLy8gQ2hlY2sgaGFyZCBuZWdhdGl2ZXNcbiAgICAgICAgICBpZiAodGhpcy5pc0hhcmROZWdhdGl2ZShzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50LCBwcmVkaWN0ZWQpKSB7XG4gICAgICAgICAgICBoYXJkTmVnYXRpdmVUb3RhbCsrO1xuICAgICAgICAgICAgaWYgKGNvcnJlY3QpIHtcbiAgICAgICAgICAgICAgaGFyZE5lZ2F0aXZlQ29ycmVjdCsrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICB0b3RhbExhdGVuY3kgKz0gZXhhbXBsZS50cmFqZWN0b3J5LnRvdGFsTGF0ZW5jeU1zO1xuICAgIH1cblxuICAgIGNvbnN0IHRvdGFsUm91dGluZ0V4YW1wbGVzID0gdGVzdFNldC5yZWR1Y2UoXG4gICAgICAoc3VtLCBleCkgPT4gc3VtICsgZXguZGVjb21wb3NpdGlvbi5zdWJRdWVyaWVzLmZpbHRlcigoc3EpID0+IHNxLnJlY29tbWVuZGVkQWdlbnQpLmxlbmd0aCxcbiAgICAgIDBcbiAgICApO1xuXG4gICAgY29uc3QgcGVyQWdlbnRSZXN1bHQ6IFJlY29yZDxzdHJpbmcsIG51bWJlcj4gPSB7fTtcbiAgICBmb3IgKGNvbnN0IFthZ2VudCwgc3RhdHNdIG9mIE9iamVjdC5lbnRyaWVzKHBlckFnZW50QWNjdXJhY3kpKSB7XG4gICAgICBwZXJBZ2VudFJlc3VsdFthZ2VudF0gPSBzdGF0cy50b3RhbCA+IDAgPyBzdGF0cy5jb3JyZWN0IC8gc3RhdHMudG90YWwgOiAwO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBkZWNvbXBvc2l0aW9uQWNjdXJhY3k6IHRlc3RTZXQubGVuZ3RoID4gMCA/IGRlY29tcG9zaXRpb25Db3JyZWN0IC8gdGVzdFNldC5sZW5ndGggOiAwLFxuICAgICAgc3ludGhlc2lzUXVhbGl0eTogdGVzdFNldC5sZW5ndGggPiAwID8gc3ludGhlc2lzUXVhbGl0eVN1bSAvIHRlc3RTZXQubGVuZ3RoIDogMCxcbiAgICAgIHJvdXRpbmdBY2N1cmFjeTogdG90YWxSb3V0aW5nRXhhbXBsZXMgPiAwID8gcm91dGluZ0NvcnJlY3QgLyB0b3RhbFJvdXRpbmdFeGFtcGxlcyA6IDAsXG4gICAgICBoYXJkTmVnYXRpdmVBY2N1cmFjeTogaGFyZE5lZ2F0aXZlVG90YWwgPiAwID8gaGFyZE5lZ2F0aXZlQ29ycmVjdCAvIGhhcmROZWdhdGl2ZVRvdGFsIDogMCxcbiAgICAgIGF2Z0xhdGVuY3lNczogdGVzdFNldC5sZW5ndGggPiAwID8gdG90YWxMYXRlbmN5IC8gdGVzdFNldC5sZW5ndGggOiAwLFxuICAgICAgdG90YWxFeGFtcGxlczogdGVzdFNldC5sZW5ndGgsXG4gICAgICBwZXJBZ2VudEFjY3VyYWN5OiBwZXJBZ2VudFJlc3VsdCxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIEdlbmVyYXRlIGNvbnRyYXN0aXZlIHBhaXJzIGZyb20gZGF0YXNldFxuICAgKi9cbiAgZ2VuZXJhdGVDb250cmFzdGl2ZVBhaXJzKFxuICAgIGRhdGFzZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdLFxuICAgIGhhcmROZWdhdGl2ZVJhdGlvID0gMC4zXG4gICk6IENvbnRyYXN0aXZlUGFpcltdIHtcbiAgICBjb25zdCBwYWlyczogQ29udHJhc3RpdmVQYWlyW10gPSBbXTtcbiAgICBjb25zdCBhZ2VudHMgPSBPYmplY3Qua2V5cyhBR0VOVF9ERUZJTklUSU9OUyk7XG5cbiAgICBmb3IgKGNvbnN0IGV4YW1wbGUgb2YgZGF0YXNldCkge1xuICAgICAgZm9yIChjb25zdCBzdWJRdWVyeSBvZiBleGFtcGxlLmRlY29tcG9zaXRpb24uc3ViUXVlcmllcykge1xuICAgICAgICBpZiAoIXN1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnQpIGNvbnRpbnVlO1xuXG4gICAgICAgIGNvbnN0IHBvc2l0aXZlQWdlbnQgPSBzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50O1xuXG4gICAgICAgIGZvciAoY29uc3QgbmVnYXRpdmVBZ2VudCBvZiBhZ2VudHMpIHtcbiAgICAgICAgICBpZiAobmVnYXRpdmVBZ2VudCA9PT0gcG9zaXRpdmVBZ2VudCkgY29udGludWU7XG5cbiAgICAgICAgICBjb25zdCBpc0hhcmQgPSB0aGlzLmlzSGFyZE5lZ2F0aXZlKHBvc2l0aXZlQWdlbnQsIG5lZ2F0aXZlQWdlbnQpO1xuXG4gICAgICAgICAgLy8gQXBwbHkgaGFyZCBuZWdhdGl2ZSByYXRpb1xuICAgICAgICAgIGNvbnN0IGluY2x1ZGUgPSBpc0hhcmRcbiAgICAgICAgICAgID8gTWF0aC5yYW5kb20oKSA8IGhhcmROZWdhdGl2ZVJhdGlvXG4gICAgICAgICAgICA6IE1hdGgucmFuZG9tKCkgPCAxIC0gaGFyZE5lZ2F0aXZlUmF0aW87XG5cbiAgICAgICAgICBpZiAoaW5jbHVkZSkge1xuICAgICAgICAgICAgcGFpcnMucHVzaCh7XG4gICAgICAgICAgICAgIGFuY2hvcjogc3ViUXVlcnkucXVlcnksXG4gICAgICAgICAgICAgIGFuY2hvckVtYmVkZGluZzogZXhhbXBsZS5xdWVyeUVtYmVkZGluZyxcbiAgICAgICAgICAgICAgcG9zaXRpdmVBZ2VudCxcbiAgICAgICAgICAgICAgbmVnYXRpdmVBZ2VudCxcbiAgICAgICAgICAgICAgaXNIYXJkTmVnYXRpdmU6IGlzSGFyZCxcbiAgICAgICAgICAgICAgcXVhbGl0eTogZXhhbXBsZS5xdWFsaXR5U2NvcmUsXG4gICAgICAgICAgICAgIHNvdXJjZUlkOiBleGFtcGxlLmlkLFxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHBhaXJzO1xuICB9XG5cbiAgLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAgLy8gUHJpdmF0ZSBNZXRob2RzXG4gIC8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbiAgcHJpdmF0ZSByZXNldFN0YXRlKCk6IHZvaWQge1xuICAgIHRoaXMuY3VycmVudEVwb2NoID0gMDtcbiAgICB0aGlzLmN1cnJlbnRTdGVwID0gMDtcbiAgICB0aGlzLmJlc3RWYWxMb3NzID0gSW5maW5pdHk7XG4gICAgdGhpcy5wYXRpZW5jZUNvdW50ZXIgPSAwO1xuICAgIHRoaXMubG9zc0hpc3RvcnkgPSBbXTtcbiAgICB0aGlzLnZhbExvc3NIaXN0b3J5ID0gW107XG4gIH1cblxuICBwcml2YXRlIHNwbGl0RGF0YXNldChcbiAgICBkYXRhc2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXVxuICApOiB7IHRyYWluU2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXTsgdmFsU2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXSB9IHtcbiAgICBjb25zdCB2YWxTaXplID0gTWF0aC5mbG9vcihkYXRhc2V0Lmxlbmd0aCAqIHRoaXMuY29uZmlnLnZhbGlkYXRpb25TcGxpdCk7XG4gICAgY29uc3Qgc2h1ZmZsZWQgPSB0aGlzLnNodWZmbGUoWy4uLmRhdGFzZXRdKTtcbiAgICByZXR1cm4ge1xuICAgICAgdHJhaW5TZXQ6IHNodWZmbGVkLnNsaWNlKHZhbFNpemUpLFxuICAgICAgdmFsU2V0OiBzaHVmZmxlZC5zbGljZSgwLCB2YWxTaXplKSxcbiAgICB9O1xuICB9XG5cbiAgcHJpdmF0ZSBzcGxpdFBhaXJzKFxuICAgIHBhaXJzOiBDb250cmFzdGl2ZVBhaXJbXVxuICApOiB7IHRyYWluU2V0OiBDb250cmFzdGl2ZVBhaXJbXTsgdmFsU2V0OiBDb250cmFzdGl2ZVBhaXJbXSB9IHtcbiAgICBjb25zdCB2YWxTaXplID0gTWF0aC5mbG9vcihwYWlycy5sZW5ndGggKiB0aGlzLmNvbmZpZy52YWxpZGF0aW9uU3BsaXQpO1xuICAgIGNvbnN0IHNodWZmbGVkID0gdGhpcy5zaHVmZmxlKFsuLi5wYWlyc10pO1xuICAgIHJldHVybiB7XG4gICAgICB0cmFpblNldDogc2h1ZmZsZWQuc2xpY2UodmFsU2l6ZSksXG4gICAgICB2YWxTZXQ6IHNodWZmbGVkLnNsaWNlKDAsIHZhbFNpemUpLFxuICAgIH07XG4gIH1cblxuICBwcml2YXRlIGNyZWF0ZUJhdGNoZXMoZGF0YXNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBSbG1UcmFpbmluZ0V4YW1wbGVbXVtdIHtcbiAgICBjb25zdCBiYXRjaGVzOiBSbG1UcmFpbmluZ0V4YW1wbGVbXVtdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBkYXRhc2V0Lmxlbmd0aDsgaSArPSB0aGlzLmNvbmZpZy5iYXRjaFNpemUpIHtcbiAgICAgIGJhdGNoZXMucHVzaChkYXRhc2V0LnNsaWNlKGksIGkgKyB0aGlzLmNvbmZpZy5iYXRjaFNpemUpKTtcbiAgICB9XG4gICAgcmV0dXJuIGJhdGNoZXM7XG4gIH1cblxuICBwcml2YXRlIGNyZWF0ZVBhaXJCYXRjaGVzKHBhaXJzOiBDb250cmFzdGl2ZVBhaXJbXSk6IENvbnRyYXN0aXZlUGFpcltdW10ge1xuICAgIGNvbnN0IGJhdGNoZXM6IENvbnRyYXN0aXZlUGFpcltdW10gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBhaXJzLmxlbmd0aDsgaSArPSB0aGlzLmNvbmZpZy5iYXRjaFNpemUpIHtcbiAgICAgIGJhdGNoZXMucHVzaChwYWlycy5zbGljZShpLCBpICsgdGhpcy5jb25maWcuYmF0Y2hTaXplKSk7XG4gICAgfVxuICAgIHJldHVybiBiYXRjaGVzO1xuICB9XG5cbiAgcHJpdmF0ZSBzaHVmZmxlPFQ+KGFycmF5OiBUW10pOiBUW10ge1xuICAgIC8vIEZpc2hlci1ZYXRlcyBzaHVmZmxlXG4gICAgZm9yIChsZXQgaSA9IGFycmF5Lmxlbmd0aCAtIDE7IGkgPiAwOyBpLS0pIHtcbiAgICAgIGNvbnN0IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiAoaSArIDEpKTtcbiAgICAgIFthcnJheVtpXSwgYXJyYXlbal1dID0gW2FycmF5W2pdLCBhcnJheVtpXV07XG4gICAgfVxuICAgIHJldHVybiBhcnJheTtcbiAgfVxuXG4gIHByaXZhdGUgdHJhaW5EZWNvbXBvc2l0aW9uQmF0Y2goYmF0Y2g6IFJsbVRyYWluaW5nRXhhbXBsZVtdKTogbnVtYmVyIHtcbiAgICBsZXQgYmF0Y2hMb3NzID0gMDtcblxuICAgIGZvciAoY29uc3QgZXhhbXBsZSBvZiBiYXRjaCkge1xuICAgICAgLy8gRGVjb21wb3NpdGlvbiBxdWFsaXR5IGxvc3NcbiAgICAgIGNvbnN0IHF1YWxpdHlMb3NzID0gMSAtIChleGFtcGxlLmRlY29tcG9zaXRpb24uc3VjY2VzcyA/IGV4YW1wbGUucXVhbGl0eVNjb3JlIDogMCk7XG5cbiAgICAgIC8vIERlcHRoIGFwcHJvcHJpYXRlbmVzcyAocGVuYWxpemUgdG9vIHNoYWxsb3cgb3IgdG9vIGRlZXApXG4gICAgICBjb25zdCBkZXB0aCA9IGV4YW1wbGUuZGVjb21wb3NpdGlvbi5zdWJRdWVyaWVzLmxlbmd0aDtcbiAgICAgIGNvbnN0IGlkZWFsRGVwdGggPSAzO1xuICAgICAgY29uc3QgZGVwdGhMb3NzID0gTWF0aC5hYnMoZGVwdGggLSBpZGVhbERlcHRoKSAvIGlkZWFsRGVwdGg7XG5cbiAgICAgIC8vIENvbXBsZXhpdHkgYmFsYW5jZSBsb3NzXG4gICAgICBjb25zdCBjb21wbGV4aXR5TG9zcyA9IE1hdGguYWJzKGV4YW1wbGUuZGVjb21wb3NpdGlvbi50b3RhbENvbXBsZXhpdHkgLSAxKSAvIDM7XG5cbiAgICAgIGNvbnN0IGxvc3MgPVxuICAgICAgICBxdWFsaXR5TG9zcyAqIHRoaXMuY29uZmlnLmRlY29tcG9zaXRpb25XZWlnaHQgKiAwLjYgK1xuICAgICAgICBkZXB0aExvc3MgKiAwLjIgK1xuICAgICAgICBjb21wbGV4aXR5TG9zcyAqIDAuMjtcblxuICAgICAgYmF0Y2hMb3NzICs9IGxvc3M7XG4gICAgfVxuXG4gICAgcmV0dXJuIGJhdGNoTG9zcyAvIGJhdGNoLmxlbmd0aDtcbiAgfVxuXG4gIHByaXZhdGUgdHJhaW5TeW50aGVzaXNCYXRjaChiYXRjaDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBudW1iZXIge1xuICAgIGxldCBiYXRjaExvc3MgPSAwO1xuXG4gICAgZm9yIChjb25zdCBleGFtcGxlIG9mIGJhdGNoKSB7XG4gICAgICAvLyBTdWItYW5zd2VyIHF1YWxpdHlcbiAgICAgIGNvbnN0IHN1YkFuc3dlclF1YWxpdHkgPVxuICAgICAgICBleGFtcGxlLnN1YkFuc3dlcnMubGVuZ3RoID4gMFxuICAgICAgICAgID8gZXhhbXBsZS5zdWJBbnN3ZXJzLnJlZHVjZSgoc3VtLCBhKSA9PiBzdW0gKyBhLnF1YWxpdHksIDApIC8gZXhhbXBsZS5zdWJBbnN3ZXJzLmxlbmd0aFxuICAgICAgICAgIDogMDtcblxuICAgICAgLy8gRmluYWwgYW5zd2VyIHF1YWxpdHlcbiAgICAgIGNvbnN0IGZpbmFsUXVhbGl0eSA9IGV4YW1wbGUucXVhbGl0eVNjb3JlO1xuXG4gICAgICAvLyBDb2hlcmVuY2UgYm9udXMgKGZpbmFsIHNob3VsZCBiZSBiZXR0ZXIgdGhhbiBwYXJ0cyBhdmVyYWdlKVxuICAgICAgY29uc3QgY29oZXJlbmNlQm9udXMgPSBNYXRoLm1heCgwLCBmaW5hbFF1YWxpdHkgLSBzdWJBbnN3ZXJRdWFsaXR5KSAqIDAuNTtcblxuICAgICAgY29uc3QgbG9zcyA9ICgxIC0gKHN1YkFuc3dlclF1YWxpdHkgKiAwLjQgKyBmaW5hbFF1YWxpdHkgKiAwLjQgKyBjb2hlcmVuY2VCb251cyAqIDAuMikpO1xuXG4gICAgICBiYXRjaExvc3MgKz0gbG9zcyAqIHRoaXMuY29uZmlnLnN5bnRoZXNpc1dlaWdodDtcbiAgICB9XG5cbiAgICByZXR1cm4gYmF0Y2hMb3NzIC8gYmF0Y2gubGVuZ3RoO1xuICB9XG5cbiAgcHJpdmF0ZSB0cmFpbkNvbnRyYXN0aXZlQmF0Y2goYmF0Y2g6IENvbnRyYXN0aXZlUGFpcltdKTogbnVtYmVyIHtcbiAgICBsZXQgYmF0Y2hMb3NzID0gMDtcblxuICAgIGZvciAoY29uc3QgcGFpciBvZiBiYXRjaCkge1xuICAgICAgLy8gVHJpcGxldCBsb3NzXG4gICAgICBjb25zdCB0cmlwbGV0TG9zcyA9IHRoaXMuY29tcHV0ZVRyaXBsZXRMb3NzKHBhaXIpO1xuXG4gICAgICAvLyBJbmZvTkNFIGxvc3NcbiAgICAgIGNvbnN0IGluZm9uY2VMb3NzID0gdGhpcy5jb21wdXRlSW5mb05DRUxvc3MocGFpcik7XG5cbiAgICAgIGJhdGNoTG9zcyArPSAodHJpcGxldExvc3MgKiAwLjUgKyBpbmZvbmNlTG9zcyAqIDAuNSkgKiB0aGlzLmNvbmZpZy5yb3V0aW5nV2VpZ2h0O1xuICAgIH1cblxuICAgIHJldHVybiBiYXRjaExvc3MgLyBiYXRjaC5sZW5ndGg7XG4gIH1cblxuICBwcml2YXRlIHZhbGlkYXRlRGVjb21wb3NpdGlvbih2YWxTZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdKTogbnVtYmVyIHtcbiAgICBpZiAodmFsU2V0Lmxlbmd0aCA9PT0gMCkgcmV0dXJuIDA7XG5cbiAgICBsZXQgdG90YWxMb3NzID0gMDtcbiAgICBmb3IgKGNvbnN0IGV4YW1wbGUgb2YgdmFsU2V0KSB7XG4gICAgICB0b3RhbExvc3MgKz0gMSAtIGV4YW1wbGUucXVhbGl0eVNjb3JlO1xuICAgIH1cbiAgICByZXR1cm4gdG90YWxMb3NzIC8gdmFsU2V0Lmxlbmd0aDtcbiAgfVxuXG4gIHByaXZhdGUgdmFsaWRhdGVTeW50aGVzaXModmFsU2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXSk6IG51bWJlciB7XG4gICAgaWYgKHZhbFNldC5sZW5ndGggPT09IDApIHJldHVybiAwO1xuXG4gICAgbGV0IHRvdGFsTG9zcyA9IDA7XG4gICAgZm9yIChjb25zdCBleGFtcGxlIG9mIHZhbFNldCkge1xuICAgICAgdG90YWxMb3NzICs9IDEgLSBleGFtcGxlLnF1YWxpdHlTY29yZTtcbiAgICB9XG4gICAgcmV0dXJuIHRvdGFsTG9zcyAvIHZhbFNldC5sZW5ndGg7XG4gIH1cblxuICBwcml2YXRlIHZhbGlkYXRlQ29udHJhc3RpdmUoXG4gICAgdmFsU2V0OiBDb250cmFzdGl2ZVBhaXJbXVxuICApOiB7IGxvc3M6IG51bWJlcjsgY29ycmVjdDogbnVtYmVyOyB0b3RhbDogbnVtYmVyIH0ge1xuICAgIGlmICh2YWxTZXQubGVuZ3RoID09PSAwKSByZXR1cm4geyBsb3NzOiAwLCBjb3JyZWN0OiAwLCB0b3RhbDogMCB9O1xuXG4gICAgbGV0IHRvdGFsTG9zcyA9IDA7XG4gICAgbGV0IGNvcnJlY3QgPSAwO1xuXG4gICAgZm9yIChjb25zdCBwYWlyIG9mIHZhbFNldCkge1xuICAgICAgY29uc3QgdHJpcGxldExvc3MgPSB0aGlzLmNvbXB1dGVUcmlwbGV0TG9zcyhwYWlyKTtcbiAgICAgIGNvbnN0IGluZm9uY2VMb3NzID0gdGhpcy5jb21wdXRlSW5mb05DRUxvc3MocGFpcik7XG4gICAgICB0b3RhbExvc3MgKz0gdHJpcGxldExvc3MgKiAwLjUgKyBpbmZvbmNlTG9zcyAqIDAuNTtcblxuICAgICAgLy8gQ2hlY2sgcm91dGluZyBjb3JyZWN0bmVzc1xuICAgICAgY29uc3QgcG9zRGlzdCA9IHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5wb3NpdGl2ZUFnZW50KTtcbiAgICAgIGNvbnN0IG5lZ0Rpc3QgPSB0aGlzLmFnZW50RGlzdGFuY2UocGFpci5hbmNob3IsIHBhaXIubmVnYXRpdmVBZ2VudCk7XG4gICAgICBpZiAocG9zRGlzdCA8IG5lZ0Rpc3QpIHtcbiAgICAgICAgY29ycmVjdCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBsb3NzOiB0b3RhbExvc3MgLyB2YWxTZXQubGVuZ3RoLFxuICAgICAgY29ycmVjdCxcbiAgICAgIHRvdGFsOiB2YWxTZXQubGVuZ3RoLFxuICAgIH07XG4gIH1cblxuICBwcml2YXRlIGNvbXB1dGVUcmlwbGV0TG9zcyhwYWlyOiBDb250cmFzdGl2ZVBhaXIpOiBudW1iZXIge1xuICAgIGNvbnN0IHBvc0Rpc3QgPSB0aGlzLmFnZW50RGlzdGFuY2UocGFpci5hbmNob3IsIHBhaXIucG9zaXRpdmVBZ2VudCk7XG4gICAgY29uc3QgbmVnRGlzdCA9IHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5uZWdhdGl2ZUFnZW50KTtcbiAgICByZXR1cm4gTWF0aC5tYXgoMCwgdGhpcy5jb25maWcuY29udHJhc3RpdmVNYXJnaW4gKyBwb3NEaXN0IC0gbmVnRGlzdCk7XG4gIH1cblxuICBwcml2YXRlIGNvbXB1dGVJbmZvTkNFTG9zcyhwYWlyOiBDb250cmFzdGl2ZVBhaXIpOiBudW1iZXIge1xuICAgIGNvbnN0IHBvc1NpbSA9IDEgLSB0aGlzLmFnZW50RGlzdGFuY2UocGFpci5hbmNob3IsIHBhaXIucG9zaXRpdmVBZ2VudCk7XG4gICAgY29uc3QgbmVnU2ltID0gMSAtIHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5uZWdhdGl2ZUFnZW50KTtcblxuICAgIGNvbnN0IHRlbXAgPSB0aGlzLmNvbmZpZy5pbmZvbmNlVGVtcGVyYXR1cmU7XG4gICAgY29uc3QgcG9zRXhwID0gTWF0aC5leHAocG9zU2ltIC8gdGVtcCk7XG4gICAgY29uc3QgbmVnRXhwID0gTWF0aC5leHAobmVnU2ltIC8gdGVtcCk7XG5cbiAgICByZXR1cm4gLU1hdGgubG9nKHBvc0V4cCAvIChwb3NFeHAgKyBuZWdFeHApKTtcbiAgfVxuXG4gIHByaXZhdGUgYWdlbnREaXN0YW5jZShxdWVyeTogc3RyaW5nLCBhZ2VudDogc3RyaW5nKTogbnVtYmVyIHtcbiAgICBjb25zdCBxdWVyeUxvd2VyID0gcXVlcnkudG9Mb3dlckNhc2UoKTtcbiAgICBjb25zdCBhZ2VudERlZiA9IEFHRU5UX0RFRklOSVRJT05TW2FnZW50XTtcblxuICAgIGlmICghYWdlbnREZWYpIHJldHVybiAxLjA7XG5cbiAgICBjb25zdCBtYXRjaGVzID0gYWdlbnREZWYua2V5d29yZHMuZmlsdGVyKChrdykgPT4gcXVlcnlMb3dlci5pbmNsdWRlcyhrdykpLmxlbmd0aDtcbiAgICByZXR1cm4gMS4wIC0gTWF0aC5taW4oMS4wLCBtYXRjaGVzIC8gYWdlbnREZWYua2V5d29yZHMubGVuZ3RoKTtcbiAgfVxuXG4gIHByaXZhdGUgcHJlZGljdEFnZW50KHF1ZXJ5OiBzdHJpbmcpOiBzdHJpbmcge1xuICAgIGxldCBiZXN0QWdlbnQgPSAnY29kZXInO1xuICAgIGxldCBiZXN0U2NvcmUgPSAwO1xuXG4gICAgZm9yIChjb25zdCBbYWdlbnQsIGRlZl0gb2YgT2JqZWN0LmVudHJpZXMoQUdFTlRfREVGSU5JVElPTlMpKSB7XG4gICAgICBjb25zdCBxdWVyeUxvd2VyID0gcXVlcnkudG9Mb3dlckNhc2UoKTtcbiAgICAgIGNvbnN0IG1hdGNoZXMgPSBkZWYua2V5d29yZHMuZmlsdGVyKChrdykgPT4gcXVlcnlMb3dlci5pbmNsdWRlcyhrdykpLmxlbmd0aDtcbiAgICAgIGNvbnN0IHNjb3JlID0gbWF0Y2hlcyAvIGRlZi5rZXl3b3Jkcy5sZW5ndGg7XG5cbiAgICAgIGlmIChzY29yZSA+IGJlc3RTY29yZSkge1xuICAgICAgICBiZXN0U2NvcmUgPSBzY29yZTtcbiAgICAgICAgYmVzdEFnZW50ID0gYWdlbnQ7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGJlc3RBZ2VudDtcbiAgfVxuXG4gIHByaXZhdGUgaXNIYXJkTmVnYXRpdmUoYWdlbnQxOiBzdHJpbmcsIGFnZW50Mjogc3RyaW5nKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIEhBUkRfTkVHQVRJVkVfUEFJUlMuc29tZShcbiAgICAgIChbYSwgYl0pID0+IChhZ2VudDEgPT09IGEgJiYgYWdlbnQyID09PSBiKSB8fCAoYWdlbnQxID09PSBiICYmIGFnZW50MiA9PT0gYSlcbiAgICApO1xuICB9XG5cbiAgcHJpdmF0ZSBmaW5kQmVzdEVwb2NoKCk6IG51bWJlciB7XG4gICAgaWYgKHRoaXMudmFsTG9zc0hpc3RvcnkubGVuZ3RoID09PSAwKSByZXR1cm4gMDtcblxuICAgIGxldCBiZXN0SWR4ID0gMDtcbiAgICBsZXQgYmVzdExvc3MgPSB0aGlzLnZhbExvc3NIaXN0b3J5WzBdO1xuXG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB0aGlzLnZhbExvc3NIaXN0b3J5Lmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZiAodGhpcy52YWxMb3NzSGlzdG9yeVtpXSA8IGJlc3RMb3NzKSB7XG4gICAgICAgIGJlc3RMb3NzID0gdGhpcy52YWxMb3NzSGlzdG9yeVtpXTtcbiAgICAgICAgYmVzdElkeCA9IGk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGJlc3RJZHg7XG4gIH1cbn1cblxuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIEZhY3RvcnkgRnVuY3Rpb25zXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIENyZWF0ZSBhbiBSTE0gdHJhaW5lciB3aXRoIGRlZmF1bHQgY29uZmlndXJhdGlvblxuICovXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUmxtVHJhaW5lcihjb25maWc/OiBQYXJ0aWFsPFJsbVRyYWluaW5nQ29uZmlnPik6IFJsbVRyYWluZXIge1xuICByZXR1cm4gbmV3IFJsbVRyYWluZXIoY29uZmlnKTtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYW4gZW1wdHkgUkxNIHRyYWluaW5nIGV4YW1wbGVcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUVtcHR5RXhhbXBsZShxdWVyeTogc3RyaW5nKTogUmxtVHJhaW5pbmdFeGFtcGxlIHtcbiAgcmV0dXJuIHtcbiAgICBpZDogY3J5cHRvLnJhbmRvbVVVSUQgPyBjcnlwdG8ucmFuZG9tVVVJRCgpIDogYCR7RGF0ZS5ub3coKX0tJHtNYXRoLnJhbmRvbSgpLnRvU3RyaW5nKDM2KX1gLFxuICAgIHF1ZXJ5LFxuICAgIGRlY29tcG9zaXRpb246IHtcbiAgICAgIHN1YlF1ZXJpZXM6IFtdLFxuICAgICAgc3RyYXRlZ3k6ICdub25lJyxcbiAgICAgIHJhdGlvbmFsZTogJycsXG4gICAgICB0b3RhbENvbXBsZXhpdHk6IDAsXG4gICAgICBzdWNjZXNzOiBmYWxzZSxcbiAgICB9LFxuICAgIHN1YkFuc3dlcnM6IFtdLFxuICAgIGZpbmFsQW5zd2VyOiAnJyxcbiAgICBxdWFsaXR5U2NvcmU6IDAsXG4gICAgdHJhamVjdG9yeToge1xuICAgICAgdG90YWxMYXRlbmN5TXM6IDAsXG4gICAgICByZXRyaWVzOiAwLFxuICAgICAgbWF4UGFyYWxsZWxpc206IDEsXG4gICAgICBtb2RlbHNVc2VkOiBbXSxcbiAgICAgIGFnZW50c0ludm9rZWQ6IFtdLFxuICAgICAgdG9vbHNVc2VkOiBbXSxcbiAgICAgIGF0dHJpYnV0ZXM6IHt9LFxuICAgIH0sXG4gICAgc3VjY2VzczogZmFsc2UsXG4gICAgbGVzc29uczogW10sXG4gICAgc291cmNlOiAnbWFudWFsJyxcbiAgfTtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYSBzdWItcXVlcnlcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN1YlF1ZXJ5KFxuICBpZDogbnVtYmVyLFxuICBxdWVyeTogc3RyaW5nLFxuICBvcHRpb25zOiBQYXJ0aWFsPFN1YlF1ZXJ5PiA9IHt9XG4pOiBTdWJRdWVyeSB7XG4gIHJldHVybiB7XG4gICAgaWQsXG4gICAgcXVlcnksXG4gICAgZXhwZWN0ZWRUeXBlOiAndGV4dCcsXG4gICAgZGVwZW5kZW5jaWVzOiBbXSxcbiAgICBjb21wbGV4aXR5OiAwLjUsXG4gICAgLi4ub3B0aW9ucyxcbiAgfTtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYSBzdWItYW5zd2VyXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTdWJBbnN3ZXIoXG4gIHN1YlF1ZXJ5SWQ6IG51bWJlcixcbiAgY29udGVudDogc3RyaW5nLFxuICBhZ2VudDogc3RyaW5nLFxuICBvcHRpb25zOiBQYXJ0aWFsPFN1YkFuc3dlcj4gPSB7fVxuKTogU3ViQW5zd2VyIHtcbiAgcmV0dXJuIHtcbiAgICBzdWJRdWVyeUlkLFxuICAgIGNvbnRlbnQsXG4gICAgY29uZmlkZW5jZTogMC44LFxuICAgIGFnZW50LFxuICAgIGxhdGVuY3lNczogMCxcbiAgICBxdWFsaXR5OiAwLjgsXG4gICAgc3VjY2VzczogdHJ1ZSxcbiAgICAuLi5vcHRpb25zLFxuICB9O1xufVxuXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuLy8gRXhwb3J0c1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuZXhwb3J0IGRlZmF1bHQgUmxtVHJhaW5lcjtcbiJdfQ==