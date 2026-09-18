"use strict";
/**
 * RLM (Recursive Learning Machine) Training Module
 *
 * Provides training capabilities for RuvLTRA models on RLM task routing
 * and decomposition, including query decomposition, answer synthesis,
 * and agent routing optimization.
 *
 * @module rlm/training
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RlmTrainer = exports.HARD_NEGATIVE_PAIRS = exports.AGENT_DEFINITIONS = exports.ROUTING_FOCUSED_CONFIG = exports.THOROUGH_RLM_CONFIG = exports.FAST_RLM_CONFIG = exports.DEFAULT_RLM_CONFIG = void 0;
exports.createRlmTrainer = createRlmTrainer;
exports.createEmptyExample = createEmptyExample;
exports.createSubQuery = createSubQuery;
exports.createSubAnswer = createSubAnswer;
// =============================================================================
// Default Configurations
// =============================================================================
/**
 * Default RLM training configuration
 */
exports.DEFAULT_RLM_CONFIG = {
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
exports.FAST_RLM_CONFIG = {
    ...exports.DEFAULT_RLM_CONFIG,
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
exports.THOROUGH_RLM_CONFIG = {
    ...exports.DEFAULT_RLM_CONFIG,
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
exports.ROUTING_FOCUSED_CONFIG = {
    ...exports.DEFAULT_RLM_CONFIG,
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
exports.AGENT_DEFINITIONS = {
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
exports.HARD_NEGATIVE_PAIRS = [
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
class RlmTrainer {
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
        this.config = { ...exports.DEFAULT_RLM_CONFIG, ...config };
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
        const agents = Object.keys(exports.AGENT_DEFINITIONS);
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
        const agentDef = exports.AGENT_DEFINITIONS[agent];
        if (!agentDef)
            return 1.0;
        const matches = agentDef.keywords.filter((kw) => queryLower.includes(kw)).length;
        return 1.0 - Math.min(1.0, matches / agentDef.keywords.length);
    }
    predictAgent(query) {
        let bestAgent = 'coder';
        let bestScore = 0;
        for (const [agent, def] of Object.entries(exports.AGENT_DEFINITIONS)) {
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
        return exports.HARD_NEGATIVE_PAIRS.some(([a, b]) => (agent1 === a && agent2 === b) || (agent1 === b && agent2 === a));
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
exports.RlmTrainer = RlmTrainer;
// =============================================================================
// Factory Functions
// =============================================================================
/**
 * Create an RLM trainer with default configuration
 */
function createRlmTrainer(config) {
    return new RlmTrainer(config);
}
/**
 * Create an empty RLM training example
 */
function createEmptyExample(query) {
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
function createSubQuery(id, query, options = {}) {
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
function createSubAnswer(subQueryId, content, agent, options = {}) {
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
exports.default = RlmTrainer;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhaW5pbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi9zcmMvcmxtL3RyYWluaW5nLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTs7Ozs7Ozs7R0FRRzs7O0FBczdCSCw0Q0FFQztBQUtELGdEQTJCQztBQUtELHdDQWFDO0FBS0QsMENBZ0JDO0FBaHhCRCxnRkFBZ0Y7QUFDaEYseUJBQXlCO0FBQ3pCLGdGQUFnRjtBQUVoRjs7R0FFRztBQUNVLFFBQUEsa0JBQWtCLEdBQXNCO0lBQ25ELGVBQWUsRUFBRSxJQUFJO0lBQ3JCLFdBQVcsRUFBRSxJQUFJO0lBQ2pCLGFBQWEsRUFBRSxJQUFJO0lBQ25CLFNBQVMsRUFBRSxFQUFFO0lBQ2IsTUFBTSxFQUFFLEVBQUU7SUFDVixpQkFBaUIsRUFBRSxHQUFHO0lBQ3RCLGtCQUFrQixFQUFFLElBQUk7SUFDeEIsbUJBQW1CLEVBQUUsR0FBRztJQUN4QixlQUFlLEVBQUUsR0FBRztJQUNwQixhQUFhLEVBQUUsR0FBRztJQUNsQixnQkFBZ0IsRUFBRSxHQUFHO0lBQ3JCLGtCQUFrQixFQUFFLENBQUM7SUFDckIsV0FBVyxFQUFFLEdBQUc7SUFDaEIscUJBQXFCLEVBQUUsQ0FBQztJQUN4QixlQUFlLEVBQUUsR0FBRztJQUNwQixJQUFJLEVBQUUsRUFBRTtDQUNULENBQUM7QUFFRjs7R0FFRztBQUNVLFFBQUEsZUFBZSxHQUFzQjtJQUNoRCxHQUFHLDBCQUFrQjtJQUNyQixNQUFNLEVBQUUsQ0FBQztJQUNULFNBQVMsRUFBRSxFQUFFO0lBQ2IsZUFBZSxFQUFFLElBQUk7SUFDckIsV0FBVyxFQUFFLElBQUk7SUFDakIsYUFBYSxFQUFFLElBQUk7SUFDbkIscUJBQXFCLEVBQUUsQ0FBQztDQUN6QixDQUFDO0FBRUY7O0dBRUc7QUFDVSxRQUFBLG1CQUFtQixHQUFzQjtJQUNwRCxHQUFHLDBCQUFrQjtJQUNyQixNQUFNLEVBQUUsRUFBRTtJQUNWLFNBQVMsRUFBRSxFQUFFO0lBQ2Is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SUFBSTtRQUNiLEdBQUcsT0FBTztLQUNYLENBQUM7QUFDSixDQUFDO0FBRUQsZ0ZBQWdGO0FBQ2hGLFVBQVU7QUFDVixnRkFBZ0Y7QUFFaEYsa0JBQWUsVUFBVSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBSTE0gKFJlY3Vyc2l2ZSBMZWFybmluZyBNYWNoaW5lKSBUcmFpbmluZyBNb2R1bGVcbiAqXG4gKiBQcm92aWRlcyB0cmFpbmluZyBjYXBhYmlsaXRpZXMgZm9yIFJ1dkxUUkEgbW9kZWxzIG9uIFJMTSB0YXNrIHJvdXRpbmdcbiAqIGFuZCBkZWNvbXBvc2l0aW9uLCBpbmNsdWRpbmcgcXVlcnkgZGVjb21wb3NpdGlvbiwgYW5zd2VyIHN5bnRoZXNpcyxcbiAqIGFuZCBhZ2VudCByb3V0aW5nIG9wdGltaXphdGlvbi5cbiAqXG4gKiBAbW9kdWxlIHJsbS90cmFpbmluZ1xuICovXG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBUeXBlcyBhbmQgSW50ZXJmYWNlc1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuLyoqXG4gKiBTdHJhdGVneSBmb3IgZGVjb21wb3NpbmcgYSBjb21wbGV4IHF1ZXJ5XG4gKi9cbmV4cG9ydCB0eXBlIERlY29tcG9zaXRpb25TdHJhdGVneSA9XG4gIHwgJ3NlcXVlbnRpYWwnXG4gIHwgJ3BhcmFsbGVsJ1xuICB8ICdoaWVyYXJjaGljYWwnXG4gIHwgJ2RhZy1iYXNlZCdcbiAgfCAnaXRlcmF0aXZlJ1xuICB8ICdub25lJztcblxuLyoqXG4gKiBBIHN1Yi1xdWVyeSBpbiB0aGUgZGVjb21wb3NpdGlvblxuICovXG5leHBvcnQgaW50ZXJmYWNlIFN1YlF1ZXJ5IHtcbiAgLyoqIFVuaXF1ZSBpZGVudGlmaWVyIHdpdGhpbiB0aGUgZGVjb21wb3NpdGlvbiAqL1xuICBpZDogbnVtYmVyO1xuICAvKiogVGhlIHN1Yi1xdWVyeSB0ZXh0ICovXG4gIHF1ZXJ5OiBzdHJpbmc7XG4gIC8qKiBFeHBlY3RlZCBvdXRwdXQgdHlwZSAoZS5nLiwgXCJjb2RlXCIsIFwiYW5hbHlzaXNcIiwgXCJkYXRhXCIpICovXG4gIGV4cGVjdGVkVHlwZTogc3RyaW5nO1xuICAvKiogRGVwZW5kZW5jaWVzIChJRHMgb2Ygc3ViLXF1ZXJpZXMgdGhhdCBtdXN0IGNvbXBsZXRlIGZpcnN0KSAqL1xuICBkZXBlbmRlbmNpZXM6IG51bWJlcltdO1xuICAvKiogUmVjb21tZW5kZWQgYWdlbnQgdHlwZSBmb3IgdGhpcyBzdWItcXVlcnkgKi9cbiAgcmVjb21tZW5kZWRBZ2VudD86IHN0cmluZztcbiAgLyoqIEVzdGltYXRlZCBjb21wbGV4aXR5ICgwLjAtMS4wKSAqL1xuICBjb21wbGV4aXR5OiBudW1iZXI7XG4gIC8qKiBPcHRpb25hbCBjb250ZXh0IGZyb20gcGFyZW50IHF1ZXJ5ICovXG4gIGNvbnRleHQ/OiBzdHJpbmc7XG59XG5cbi8qKlxuICogRGVjb21wb3NpdGlvbiBvZiBhIGNvbXBsZXggcXVlcnkgaW50byBzdWItcXVlcmllc1xuICovXG5leHBvcnQgaW50ZXJmYWNlIFF1ZXJ5RGVjb21wb3NpdGlvbiB7XG4gIC8qKiBTdWItcXVlcmllcyBpbiBleGVjdXRpb24gb3JkZXIgKi9cbiAgc3ViUXVlcmllczogU3ViUXVlcnlbXTtcbiAgLyoqIERlY29tcG9zaXRpb24gc3RyYXRlZ3kgdXNlZCAqL1xuICBzdHJhdGVneTogRGVjb21wb3NpdGlvblN0cmF0ZWd5O1xuICAvKiogUmVhc29uaW5nIGZvciB0aGlzIGRlY29tcG9zaXRpb24gKi9cbiAgcmF0aW9uYWxlOiBzdHJpbmc7XG4gIC8qKiBUb3RhbCBlc3RpbWF0ZWQgY29tcGxleGl0eSAqL1xuICB0b3RhbENvbXBsZXhpdHk6IG51bWJlcjtcbiAgLyoqIFdoZXRoZXIgZGVjb21wb3NpdGlvbiB3YXMgc3VjY2Vzc2Z1bCAqL1xuICBzdWNjZXNzOiBib29sZWFuO1xuICAvKiogRXJyb3IgbWVzc2FnZSBpZiBkZWNvbXBvc2l0aW9uIGZhaWxlZCAqL1xuICBlcnJvcj86IHN0cmluZztcbn1cblxuLyoqXG4gKiBBbnN3ZXIgdG8gYSBzdWItcXVlcnlcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBTdWJBbnN3ZXIge1xuICAvKiogSUQgb2YgdGhlIHN1Yi1xdWVyeSB0aGlzIGFuc3dlcnMgKi9cbiAgc3ViUXVlcnlJZDogbnVtYmVyO1xuICAvKiogVGhlIGFuc3dlciBjb250ZW50ICovXG4gIGNvbnRlbnQ6IHN0cmluZztcbiAgLyoqIENvbmZpZGVuY2UgaW4gdGhpcyBhbnN3ZXIgKDAuMC0xLjApICovXG4gIGNvbmZpZGVuY2U6IG51bWJlcjtcbiAgLyoqIEFnZW50IHRoYXQgcHJvZHVjZWQgdGhpcyBhbnN3ZXIgKi9cbiAgYWdlbnQ6IHN0cmluZztcbiAgLyoqIExhdGVuY3kgaW4gbWlsbGlzZWNvbmRzICovXG4gIGxhdGVuY3lNczogbnVtYmVyO1xuICAvKiogUXVhbGl0eSBzY29yZSAoMC4wLTEuMCkgKi9cbiAgcXVhbGl0eTogbnVtYmVyO1xuICAvKiogV2hldGhlciB0aGlzIGFuc3dlciB3YXMgc3VjY2Vzc2Z1bCAqL1xuICBzdWNjZXNzOiBib29sZWFuO1xuICAvKiogRXJyb3IgbWVzc2FnZSBpZiBmYWlsZWQgKi9cbiAgZXJyb3I/OiBzdHJpbmc7XG4gIC8qKiBJbnRlcm1lZGlhdGUgcmVhc29uaW5nL2NoYWluLW9mLXRob3VnaHQgKi9cbiAgcmVhc29uaW5nPzogc3RyaW5nO1xufVxuXG4vKipcbiAqIE1ldGFkYXRhIGFib3V0IHRoZSBSTE0gZXhlY3V0aW9uIHRyYWplY3RvcnlcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBSbG1UcmFqZWN0b3J5TWV0YWRhdGEge1xuICAvKiogU2Vzc2lvbiBJRCAqL1xuICBzZXNzaW9uSWQ/OiBzdHJpbmc7XG4gIC8qKiBVc2VyIElEICovXG4gIHVzZXJJZD86IHN0cmluZztcbiAgLyoqIFRvdGFsIGxhdGVuY3kgaW4gbWlsbGlzZWNvbmRzICovXG4gIHRvdGFsTGF0ZW5jeU1zOiBudW1iZXI7XG4gIC8qKiBOdW1iZXIgb2YgcmV0cmllcyAqL1xuICByZXRyaWVzOiBudW1iZXI7XG4gIC8qKiBNYXhpbXVtIHBhcmFsbGVsIGJyYW5jaGVzIGV4ZWN1dGVkICovXG4gIG1heFBhcmFsbGVsaXNtOiBudW1iZXI7XG4gIC8qKiBNb2RlbHMgdXNlZCBkdXJpbmcgZXhlY3V0aW9uICovXG4gIG1vZGVsc1VzZWQ6IHN0cmluZ1tdO1xuICAvKiogQWdlbnRzIGludm9rZWQgKi9cbiAgYWdlbnRzSW52b2tlZDogc3RyaW5nW107XG4gIC8qKiBUb29scyB1c2VkICovXG4gIHRvb2xzVXNlZDogc3RyaW5nW107XG4gIC8qKiBDdXN0b20gYXR0cmlidXRlcyAqL1xuICBhdHRyaWJ1dGVzOiBSZWNvcmQ8c3RyaW5nLCBzdHJpbmc+O1xufVxuXG4vKipcbiAqIEEgY29tcGxldGUgUkxNIHRyYWluaW5nIGV4YW1wbGVcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBSbG1UcmFpbmluZ0V4YW1wbGUge1xuICAvKiogVW5pcXVlIGlkZW50aWZpZXIgKi9cbiAgaWQ6IHN0cmluZztcbiAgLyoqIE9yaWdpbmFsIGNvbXBsZXggcXVlcnkgKi9cbiAgcXVlcnk6IHN0cmluZztcbiAgLyoqIFF1ZXJ5IGVtYmVkZGluZyAob3B0aW9uYWwpICovXG4gIHF1ZXJ5RW1iZWRkaW5nPzogbnVtYmVyW107XG4gIC8qKiBIb3cgdGhlIHF1ZXJ5IHdhcyBkZWNvbXBvc2VkICovXG4gIGRlY29tcG9zaXRpb246IFF1ZXJ5RGVjb21wb3NpdGlvbjtcbiAgLyoqIEFuc3dlcnMgdG8gZWFjaCBzdWItcXVlcnkgKi9cbiAgc3ViQW5zd2VyczogU3ViQW5zd2VyW107XG4gIC8qKiBGaW5hbCBzeW50aGVzaXplZCBhbnN3ZXIgKi9cbiAgZmluYWxBbnN3ZXI6IHN0cmluZztcbiAgLyoqIEZpbmFsIGFuc3dlciBlbWJlZGRpbmcgKG9wdGlvbmFsKSAqL1xuICBmaW5hbEVtYmVkZGluZz86IG51bWJlcltdO1xuICAvKiogT3ZlcmFsbCBxdWFsaXR5IHNjb3JlICgwLjAtMS4wKSAqL1xuICBxdWFsaXR5U2NvcmU6IG51bWJlcjtcbiAgLyoqIEV4ZWN1dGlvbiB0cmFqZWN0b3J5IG1ldGFkYXRhICovXG4gIHRyYWplY3Rvcnk6IFJsbVRyYWplY3RvcnlNZXRhZGF0YTtcbiAgLyoqIFdoZXRoZXIgdGhpcyBleGFtcGxlIHdhcyBzdWNjZXNzZnVsICovXG4gIHN1Y2Nlc3M6IGJvb2xlYW47XG4gIC8qKiBMZXNzb25zIGxlYXJuZWQgZnJvbSB0aGlzIGV4YW1wbGUgKi9cbiAgbGVzc29uczogc3RyaW5nW107XG4gIC8qKiBTb3VyY2Ugb2YgdGhpcyBleGFtcGxlICovXG4gIHNvdXJjZTogc3RyaW5nO1xufVxuXG4vKipcbiAqIEEgY29udHJhc3RpdmUgcGFpciBmb3IgYWdlbnQgcm91dGluZyB0cmFpbmluZ1xuICovXG5leHBvcnQgaW50ZXJmYWNlIENvbnRyYXN0aXZlUGFpciB7XG4gIC8qKiBBbmNob3IgcXVlcnkgKi9cbiAgYW5jaG9yOiBzdHJpbmc7XG4gIC8qKiBBbmNob3IgZW1iZWRkaW5nIChvcHRpb25hbCkgKi9cbiAgYW5jaG9yRW1iZWRkaW5nPzogbnVtYmVyW107XG4gIC8qKiBQb3NpdGl2ZSBhZ2VudCAoY29ycmVjdCByb3V0aW5nKSAqL1xuICBwb3NpdGl2ZUFnZW50OiBzdHJpbmc7XG4gIC8qKiBOZWdhdGl2ZSBhZ2VudCAoaW5jb3JyZWN0IHJvdXRpbmcpICovXG4gIG5lZ2F0aXZlQWdlbnQ6IHN0cmluZztcbiAgLyoqIFdoZXRoZXIgdGhpcyBpcyBhIGhhcmQgbmVnYXRpdmUgKi9cbiAgaXNIYXJkTmVnYXRpdmU6IGJvb2xlYW47XG4gIC8qKiBRdWFsaXR5IHNjb3JlIG9mIHRoZSBhbmNob3IgZXhhbXBsZSAqL1xuICBxdWFsaXR5OiBudW1iZXI7XG4gIC8qKiBTb3VyY2UgZXhhbXBsZSBJRCAqL1xuICBzb3VyY2VJZDogc3RyaW5nO1xufVxuXG4vKipcbiAqIENvbmZpZ3VyYXRpb24gZm9yIFJMTSB0cmFpbmluZ1xuICovXG5leHBvcnQgaW50ZXJmYWNlIFJsbVRyYWluaW5nQ29uZmlnIHtcbiAgLyoqIExlYXJuaW5nIHJhdGUgZm9yIGRlY29tcG9zaXRpb24gdHJhaW5pbmcgKi9cbiAgZGVjb21wb3NpdGlvbkxyOiBudW1iZXI7XG4gIC8qKiBMZWFybmluZyByYXRlIGZvciBzeW50aGVzaXMgdHJhaW5pbmcgKi9cbiAgc3ludGhlc2lzTHI6IG51bWJlcjtcbiAgLyoqIExlYXJuaW5nIHJhdGUgZm9yIGNvbnRyYXN0aXZlIGZpbmUtdHVuaW5nICovXG4gIGNvbnRyYXN0aXZlTHI6IG51bWJlcjtcbiAgLyoqIEJhdGNoIHNpemUgKi9cbiAgYmF0Y2hTaXplOiBudW1iZXI7XG4gIC8qKiBOdW1iZXIgb2YgZXBvY2hzICovXG4gIGVwb2NoczogbnVtYmVyO1xuICAvKiogQ29udHJhc3RpdmUgbWFyZ2luIGZvciB0cmlwbGV0IGxvc3MgKi9cbiAgY29udHJhc3RpdmVNYXJnaW46IG51bWJlcjtcbiAgLyoqIFRlbXBlcmF0dXJlIGZvciBJbmZvTkNFIGxvc3MgKi9cbiAgaW5mb25jZVRlbXBlcmF0dXJlOiBudW1iZXI7XG4gIC8qKiBXZWlnaHQgZm9yIGRlY29tcG9zaXRpb24gbG9zcyAqL1xuICBkZWNvbXBvc2l0aW9uV2VpZ2h0OiBudW1iZXI7XG4gIC8qKiBXZWlnaHQgZm9yIHN5bnRoZXNpcyBsb3NzICovXG4gIHN5bnRoZXNpc1dlaWdodDogbnVtYmVyO1xuICAvKiogV2VpZ2h0IGZvciByb3V0aW5nIGxvc3MgKi9cbiAgcm91dGluZ1dlaWdodDogbnVtYmVyO1xuICAvKiogTWluaW11bSBxdWFsaXR5IGZvciB1cGRhdGVzICovXG4gIHF1YWxpdHlUaHJlc2hvbGQ6IG51bWJlcjtcbiAgLyoqIEV2YWx1YXRpb24gaW50ZXJ2YWwgKGVwb2NocykgKi9cbiAgZXZhbHVhdGlvbkludGVydmFsOiBudW1iZXI7XG4gIC8qKiBXYXJtdXAgc3RlcHMgKi9cbiAgd2FybXVwU3RlcHM6IG51bWJlcjtcbiAgLyoqIEVhcmx5IHN0b3BwaW5nIHBhdGllbmNlICovXG4gIGVhcmx5U3RvcHBpbmdQYXRpZW5jZTogbnVtYmVyO1xuICAvKiogVmFsaWRhdGlvbiBzcGxpdCByYXRpbyAqL1xuICB2YWxpZGF0aW9uU3BsaXQ6IG51bWJlcjtcbiAgLyoqIFJhbmRvbSBzZWVkICovXG4gIHNlZWQ6IG51bWJlcjtcbn1cblxuLyoqXG4gKiBUcmFpbmluZyByZXN1bHQgZm9yIGEgcGhhc2VcbiAqL1xuZXhwb3J0IGludGVyZmFjZSBUcmFpbmluZ1Jlc3VsdCB7XG4gIC8qKiBUcmFpbmluZyBwaGFzZSBuYW1lICovXG4gIHBoYXNlOiBzdHJpbmc7XG4gIC8qKiBFcG9jaHMgY29tcGxldGVkICovXG4gIGVwb2Noc0NvbXBsZXRlZDogbnVtYmVyO1xuICAvKiogVG90YWwgc3RlcHMgKi9cbiAgdG90YWxTdGVwczogbnVtYmVyO1xuICAvKiogRmluYWwgdHJhaW5pbmcgbG9zcyAqL1xuICBmaW5hbExvc3M6IG51bWJlcjtcbiAgLyoqIEJlc3QgdmFsaWRhdGlvbiBsb3NzICovXG4gIGJlc3RWYWxMb3NzOiBudW1iZXI7XG4gIC8qKiBCZXN0IGVwb2NoICovXG4gIGJlc3RFcG9jaDogbnVtYmVyO1xuICAvKiogRmluYWwgYWNjdXJhY3kgKGZvciBjbGFzc2lmaWNhdGlvbiB0YXNrcykgKi9cbiAgYWNjdXJhY3k6IG51bWJlcjtcbiAgLyoqIExvc3MgaGlzdG9yeSBwZXIgZXBvY2ggKi9cbiAgbG9zc0hpc3Rvcnk6IG51bWJlcltdO1xuICAvKiogVmFsaWRhdGlvbiBsb3NzIGhpc3RvcnkgKi9cbiAgdmFsTG9zc0hpc3Rvcnk6IG51bWJlcltdO1xuICAvKiogVHJhaW5pbmcgZHVyYXRpb24gaW4gbWlsbGlzZWNvbmRzICovXG4gIGR1cmF0aW9uTXM6IG51bWJlcjtcbiAgLyoqIFdoZXRoZXIgZWFybHkgc3RvcHBpbmcgd2FzIHRyaWdnZXJlZCAqL1xuICBlYXJseVN0b3BwZWQ6IGJvb2xlYW47XG59XG5cbi8qKlxuICogRXZhbHVhdGlvbiByZXN1bHQgZm9yIHRoZSB0cmFpbmVkIG1vZGVsXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgRXZhbHVhdGlvblJlc3VsdCB7XG4gIC8qKiBEZWNvbXBvc2l0aW9uIGFjY3VyYWN5ICovXG4gIGRlY29tcG9zaXRpb25BY2N1cmFjeTogbnVtYmVyO1xuICAvKiogU3ludGhlc2lzIHF1YWxpdHkgKi9cbiAgc3ludGhlc2lzUXVhbGl0eTogbnVtYmVyO1xuICAvKiogUm91dGluZyBhY2N1cmFjeSAqL1xuICByb3V0aW5nQWNjdXJhY3k6IG51bWJlcjtcbiAgLyoqIEhhcmQgbmVnYXRpdmUgYWNjdXJhY3kgKi9cbiAgaGFyZE5lZ2F0aXZlQWNjdXJhY3k6IG51bWJlcjtcbiAgLyoqIEF2ZXJhZ2UgbGF0ZW5jeSBpbiBtcyAqL1xuICBhdmdMYXRlbmN5TXM6IG51bWJlcjtcbiAgLyoqIFRvdGFsIGV4YW1wbGVzIGV2YWx1YXRlZCAqL1xuICB0b3RhbEV4YW1wbGVzOiBudW1iZXI7XG4gIC8qKiBQZXItYWdlbnQgYWNjdXJhY3kgKi9cbiAgcGVyQWdlbnRBY2N1cmFjeTogUmVjb3JkPHN0cmluZywgbnVtYmVyPjtcbn1cblxuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIERlZmF1bHQgQ29uZmlndXJhdGlvbnNcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8qKlxuICogRGVmYXVsdCBSTE0gdHJhaW5pbmcgY29uZmlndXJhdGlvblxuICovXG5leHBvcnQgY29uc3QgREVGQVVMVF9STE1fQ09ORklHOiBSbG1UcmFpbmluZ0NvbmZpZyA9IHtcbiAgZGVjb21wb3NpdGlvbkxyOiAxZS01LFxuICBzeW50aGVzaXNMcjogMWUtNSxcbiAgY29udHJhc3RpdmVMcjogMmUtNSxcbiAgYmF0Y2hTaXplOiAzMixcbiAgZXBvY2hzOiAxMCxcbiAgY29udHJhc3RpdmVNYXJnaW46IDAuNSxcbiAgaW5mb25jZVRlbXBlcmF0dXJlOiAwLjA3LFxuICBkZWNvbXBvc2l0aW9uV2VpZ2h0OiAxLjAsXG4gIHN5bnRoZXNpc1dlaWdodDogMS4wLFxuICByb3V0aW5nV2VpZ2h0OiAxLjAsXG4gIHF1YWxpdHlUaHJlc2hvbGQ6IDAuNyxcbiAgZXZhbHVhdGlvbkludGVydmFsOiAxLFxuICB3YXJtdXBTdGVwczogMTAwLFxuICBlYXJseVN0b3BwaW5nUGF0aWVuY2U6IDMsXG4gIHZhbGlkYXRpb25TcGxpdDogMC4xLFxuICBzZWVkOiA0Mixcbn07XG5cbi8qKlxuICogRmFzdCB0cmFpbmluZyBjb25maWd1cmF0aW9uXG4gKi9cbmV4cG9ydCBjb25zdCBGQVNUX1JMTV9DT05GSUc6IFJsbVRyYWluaW5nQ29uZmlnID0ge1xuICAuLi5ERUZBVUxUX1JMTV9DT05GSUcsXG4gIGVwb2NoczogMyxcbiAgYmF0Y2hTaXplOiA2NCxcbiAgZGVjb21wb3NpdGlvbkxyOiAxZS00LFxuICBzeW50aGVzaXNMcjogMWUtNCxcbiAgY29udHJhc3RpdmVMcjogNWUtNSxcbiAgZWFybHlTdG9wcGluZ1BhdGllbmNlOiAxLFxufTtcblxuLyoqXG4gKiBUaG9yb3VnaCB0cmFpbmluZyBjb25maWd1cmF0aW9uXG4gKi9cbmV4cG9ydCBjb25zdCBUSE9ST1VHSF9STE1fQ09ORklHOiBSbG1UcmFpbmluZ0NvbmZpZyA9IHtcbiAgLi4uREVGQVVMVF9STE1fQ09ORklHLFxuICBlcG9jaHM6IDUwLFxuICBiYXRjaFNpemU6IDE2LFxuICBkZWNvbXBvc2l0aW9uTHI6IDVlLTYsXG4gIHN5bnRoZXNpc0xyOiA1ZS02LFxuICBjb250cmFzdGl2ZUxyOiAxZS01LFxuICBlYXJseVN0b3BwaW5nUGF0aWVuY2U6IDEwLFxufTtcblxuLyoqXG4gKiBSb3V0aW5nLWZvY3VzZWQgdHJhaW5pbmcgY29uZmlndXJhdGlvblxuICovXG5leHBvcnQgY29uc3QgUk9VVElOR19GT0NVU0VEX0NPTkZJRzogUmxtVHJhaW5pbmdDb25maWcgPSB7XG4gIC4uLkRFRkFVTFRfUkxNX0NPTkZJRyxcbiAgcm91dGluZ1dlaWdodDogMi4wLFxuICBkZWNvbXBvc2l0aW9uV2VpZ2h0OiAwLjUsXG4gIHN5bnRoZXNpc1dlaWdodDogMC41LFxuICBjb250cmFzdGl2ZUxyOiAzZS01LFxuICBjb250cmFzdGl2ZU1hcmdpbjogMC4zLFxuICBpbmZvbmNlVGVtcGVyYXR1cmU6IDAuMDUsXG59O1xuXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuLy8gQWdlbnQgRGVmaW5pdGlvbnNcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8qKlxuICogQWdlbnQgdHlwZXMgd2l0aCBkZXNjcmlwdGlvbnMgYW5kIGtleXdvcmRzXG4gKi9cbmV4cG9ydCBjb25zdCBBR0VOVF9ERUZJTklUSU9OUzogUmVjb3JkPHN0cmluZywgeyBkZXNjcmlwdGlvbjogc3RyaW5nOyBrZXl3b3Jkczogc3RyaW5nW10gfT4gPSB7XG4gIGNvZGVyOiB7XG4gICAgZGVzY3JpcHRpb246ICdTb2Z0d2FyZSBkZXZlbG9wZXIgd2hvIHdyaXRlcyBhbmQgaW1wbGVtZW50cyBjb2RlJyxcbiAgICBrZXl3b3JkczogWydpbXBsZW1lbnQnLCAnYnVpbGQnLCAnY3JlYXRlJywgJ2NvZGUnLCAnd3JpdGUnLCAnZGV2ZWxvcCcsICdwcm9ncmFtJ10sXG4gIH0sXG4gIHJlc2VhcmNoZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ1RlY2huaWNhbCByZXNlYXJjaGVyIHdobyBpbnZlc3RpZ2F0ZXMgYW5kIGFuYWx5emVzJyxcbiAgICBrZXl3b3JkczogWydyZXNlYXJjaCcsICdpbnZlc3RpZ2F0ZScsICdhbmFseXplJywgJ2V4cGxvcmUnLCAnc3R1ZHknLCAnZXhhbWluZSddLFxuICB9LFxuICByZXZpZXdlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnQ29kZSByZXZpZXdlciB3aG8gZXZhbHVhdGVzIGNvZGUgcXVhbGl0eScsXG4gICAga2V5d29yZHM6IFsncmV2aWV3JywgJ2NoZWNrJywgJ2V2YWx1YXRlJywgJ2Fzc2VzcycsICdleGFtaW5lJywgJ2luc3BlY3QnXSxcbiAgfSxcbiAgdGVzdGVyOiB7XG4gICAgZGVzY3JpcHRpb246ICdRQSBlbmdpbmVlciB3aG8gd3JpdGVzIGFuZCBydW5zIHRlc3RzJyxcbiAgICBrZXl3b3JkczogWyd0ZXN0JywgJ3VuaXQgdGVzdCcsICdjb3ZlcmFnZScsICd2YWxpZGF0ZScsICd2ZXJpZnknLCAncWEnXSxcbiAgfSxcbiAgYXJjaGl0ZWN0OiB7XG4gICAgZGVzY3JpcHRpb246ICdTeXN0ZW0gYXJjaGl0ZWN0IHdobyBkZXNpZ25zIHNvZnR3YXJlIHN0cnVjdHVyZScsXG4gICAga2V5d29yZHM6IFsnZGVzaWduJywgJ3BsYW4nLCAnYXJjaGl0ZWN0dXJlJywgJ3NjaGVtYScsICdzdHJ1Y3R1cmUnLCAnZGlhZ3JhbSddLFxuICB9LFxuICAnc2VjdXJpdHktYXJjaGl0ZWN0Jzoge1xuICAgIGRlc2NyaXB0aW9uOiAnU2VjdXJpdHkgc3BlY2lhbGlzdCB3aG8gYXVkaXRzIHZ1bG5lcmFiaWxpdGllcycsXG4gICAga2V5d29yZHM6IFsnc2VjdXJpdHknLCAnYXVkaXQnLCAndnVsbmVyYWJpbGl0eScsICd4c3MnLCAnaW5qZWN0aW9uJywgJ2N2ZSddLFxuICB9LFxuICBkZWJ1Z2dlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnQnVnIGh1bnRlciB3aG8gZml4ZXMgZXJyb3JzIGFuZCB0cmFjZXMgaXNzdWVzJyxcbiAgICBrZXl3b3JkczogWydmaXgnLCAnZGVidWcnLCAnYnVnJywgJ2Vycm9yJywgJ3RyYWNlJywgJ2NyYXNoJywgJ3Ryb3VibGVzaG9vdCddLFxuICB9LFxuICBkb2N1bWVudGVyOiB7XG4gICAgZGVzY3JpcHRpb246ICdUZWNobmljYWwgd3JpdGVyIHdobyBjcmVhdGVzIGRvY3VtZW50YXRpb24nLFxuICAgIGtleXdvcmRzOiBbJ2RvY3VtZW50JywgJ2pzZG9jJywgJ3JlYWRtZScsICdjb21tZW50JywgJ2V4cGxhaW4nLCAnZGVzY3JpYmUnXSxcbiAgfSxcbiAgcmVmYWN0b3Jlcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnQ29kZSBtb2Rlcm5pemVyIHdobyByZXN0cnVjdHVyZXMgd2l0aG91dCBjaGFuZ2luZyBiZWhhdmlvcicsXG4gICAga2V5d29yZHM6IFsncmVmYWN0b3InLCAncmVzdHJ1Y3R1cmUnLCAnbW9kZXJuaXplJywgJ2NsZWFuJywgJ3NpbXBsaWZ5JywgJ2NvbnNvbGlkYXRlJ10sXG4gIH0sXG4gIG9wdGltaXplcjoge1xuICAgIGRlc2NyaXB0aW9uOiAnUGVyZm9ybWFuY2UgZW5naW5lZXIgd2hvIHNwZWVkcyB1cCBzbG93IGNvZGUnLFxuICAgIGtleXdvcmRzOiBbJ29wdGltaXplJywgJ3BlcmZvcm1hbmNlJywgJ3NwZWVkJywgJ2NhY2hlJywgJ2ltcHJvdmUnLCAnZmFzdGVyJ10sXG4gIH0sXG4gIGRldm9wczoge1xuICAgIGRlc2NyaXB0aW9uOiAnRGV2T3BzIGVuZ2luZWVyIHdobyBtYW5hZ2VzIGRlcGxveW1lbnQgYW5kIGluZnJhc3RydWN0dXJlJyxcbiAgICBrZXl3b3JkczogWydkZXBsb3knLCAnY2kvY2QnLCAna3ViZXJuZXRlcycsICdkb2NrZXInLCAnaW5mcmFzdHJ1Y3R1cmUnLCAncGlwZWxpbmUnXSxcbiAgfSxcbiAgJ2FwaS1kb2NzJzoge1xuICAgIGRlc2NyaXB0aW9uOiAnQVBJIGRvY3VtZW50YXRpb24gc3BlY2lhbGlzdCB3aG8gY3JlYXRlcyBzcGVjcycsXG4gICAga2V5d29yZHM6IFsnb3BlbmFwaScsICdzd2FnZ2VyJywgJ2FwaSByZWZlcmVuY2UnLCAnZW5kcG9pbnQnLCAnc3BlYycsICdyZXN0J10sXG4gIH0sXG4gIHBsYW5uZXI6IHtcbiAgICBkZXNjcmlwdGlvbjogJ1Byb2plY3QgcGxhbm5lciB3aG8gb3JnYW5pemVzIGFuZCBzY2hlZHVsZXMgd29yaycsXG4gICAga2V5d29yZHM6IFsncGxhbicsICdlc3RpbWF0ZScsICdzY2hlZHVsZScsICd0aW1lbGluZScsICdzcHJpbnQnLCAncm9hZG1hcCddLFxuICB9LFxufTtcblxuLyoqXG4gKiBIYXJkIG5lZ2F0aXZlIHBhaXJzIChjb25mdXNhYmxlIGFnZW50IGNvbWJpbmF0aW9ucylcbiAqL1xuZXhwb3J0IGNvbnN0IEhBUkRfTkVHQVRJVkVfUEFJUlM6IFtzdHJpbmcsIHN0cmluZ11bXSA9IFtcbiAgWydjb2RlcicsICdkZWJ1Z2dlciddLFxuICBbJ2NvZGVyJywgJ3JlZmFjdG9yZXInXSxcbiAgWydyZXNlYXJjaGVyJywgJ3Jldmlld2VyJ10sXG4gIFsndGVzdGVyJywgJ3Jldmlld2VyJ10sXG4gIFsnYXJjaGl0ZWN0JywgJ3BsYW5uZXInXSxcbiAgWydkb2N1bWVudGVyJywgJ2FwaS1kb2NzJ10sXG4gIFsnb3B0aW1pemVyJywgJ2RlYnVnZ2VyJ10sXG4gIFsnZGV2b3BzJywgJ2FyY2hpdGVjdCddLFxuICBbJ3NlY3VyaXR5LWFyY2hpdGVjdCcsICdyZXZpZXdlciddLFxuXTtcblxuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIFJMTSBUcmFpbmVyIENsYXNzXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIFJMTSBUcmFpbmVyIGZvciBSdXZMVFJBIG1vZGVsc1xuICpcbiAqIFByb3ZpZGVzIHRyYWluaW5nIGNhcGFiaWxpdGllcyBmb3IgZGVjb21wb3NpdGlvbiwgc3ludGhlc2lzLCBhbmQgcm91dGluZyB0YXNrcy5cbiAqL1xuZXhwb3J0IGNsYXNzIFJsbVRyYWluZXIge1xuICBwcml2YXRlIGNvbmZpZzogUmxtVHJhaW5pbmdDb25maWc7XG4gIHByaXZhdGUgY3VycmVudEVwb2NoID0gMDtcbiAgcHJpdmF0ZSBjdXJyZW50U3RlcCA9IDA7XG4gIHByaXZhdGUgYmVzdFZhbExvc3MgPSBJbmZpbml0eTtcbiAgcHJpdmF0ZSBwYXRpZW5jZUNvdW50ZXIgPSAwO1xuICBwcml2YXRlIGxvc3NIaXN0b3J5OiBudW1iZXJbXSA9IFtdO1xuICBwcml2YXRlIHZhbExvc3NIaXN0b3J5OiBudW1iZXJbXSA9IFtdO1xuXG4gIC8qKlxuICAgKiBDcmVhdGUgYSBuZXcgUkxNIHRyYWluZXJcbiAgICovXG4gIGNvbnN0cnVjdG9yKGNvbmZpZzogUGFydGlhbDxSbG1UcmFpbmluZ0NvbmZpZz4gPSB7fSkge1xuICAgIHRoaXMuY29uZmlnID0geyAuLi5ERUZBVUxUX1JMTV9DT05GSUcsIC4uLmNvbmZpZyB9O1xuICB9XG5cbiAgLyoqXG4gICAqIFRyYWluIG9uIGRlY29tcG9zaXRpb24gdGFza1xuICAgKlxuICAgKiBMZWFybnMgdG8gYnJlYWsgY29tcGxleCBxdWVyaWVzIGludG8gbWFuYWdlYWJsZSBzdWItcXVlcmllcy5cbiAgICovXG4gIGFzeW5jIHRyYWluRGVjb21wb3NpdGlvbihkYXRhc2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXSk6IFByb21pc2U8VHJhaW5pbmdSZXN1bHQ+IHtcbiAgICBjb25zdCBzdGFydFRpbWUgPSBEYXRlLm5vdygpO1xuICAgIHRoaXMucmVzZXRTdGF0ZSgpO1xuXG4gICAgY29uc3QgeyB0cmFpblNldCwgdmFsU2V0IH0gPSB0aGlzLnNwbGl0RGF0YXNldChkYXRhc2V0KTtcbiAgICBjb25zdCBiYXRjaGVzID0gdGhpcy5jcmVhdGVCYXRjaGVzKHRyYWluU2V0KTtcblxuICAgIGZvciAobGV0IGVwb2NoID0gMDsgZXBvY2ggPCB0aGlzLmNvbmZpZy5lcG9jaHM7IGVwb2NoKyspIHtcbiAgICAgIHRoaXMuY3VycmVudEVwb2NoID0gZXBvY2g7XG4gICAgICBsZXQgZXBvY2hMb3NzID0gMDtcblxuICAgICAgZm9yIChjb25zdCBiYXRjaCBvZiBiYXRjaGVzKSB7XG4gICAgICAgIGNvbnN0IGJhdGNoTG9zcyA9IHRoaXMudHJhaW5EZWNvbXBvc2l0aW9uQmF0Y2goYmF0Y2gpO1xuICAgICAgICBlcG9jaExvc3MgKz0gYmF0Y2hMb3NzO1xuICAgICAgICB0aGlzLmN1cnJlbnRTdGVwKys7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGF2Z0xvc3MgPSBlcG9jaExvc3MgLyBiYXRjaGVzLmxlbmd0aDtcbiAgICAgIHRoaXMubG9zc0hpc3RvcnkucHVzaChhdmdMb3NzKTtcblxuICAgICAgLy8gVmFsaWRhdGlvblxuICAgICAgY29uc3QgdmFsTG9zcyA9IHRoaXMudmFsaWRhdGVEZWNvbXBvc2l0aW9uKHZhbFNldCk7XG4gICAgICB0aGlzLnZhbExvc3NIaXN0b3J5LnB1c2godmFsTG9zcyk7XG5cbiAgICAgIC8vIEVhcmx5IHN0b3BwaW5nXG4gICAgICBpZiAodmFsTG9zcyA8IHRoaXMuYmVzdFZhbExvc3MpIHtcbiAgICAgICAgdGhpcy5iZXN0VmFsTG9zcyA9IHZhbExvc3M7XG4gICAgICAgIHRoaXMucGF0aWVuY2VDb3VudGVyID0gMDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMucGF0aWVuY2VDb3VudGVyKys7XG4gICAgICAgIGlmICh0aGlzLnBhdGllbmNlQ291bnRlciA+PSB0aGlzLmNvbmZpZy5lYXJseVN0b3BwaW5nUGF0aWVuY2UpIHtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBwaGFzZTogJ2RlY29tcG9zaXRpb24nLFxuICAgICAgZXBvY2hzQ29tcGxldGVkOiB0aGlzLmN1cnJlbnRFcG9jaCArIDEsXG4gICAgICB0b3RhbFN0ZXBzOiB0aGlzLmN1cnJlbnRTdGVwLFxuICAgICAgZmluYWxMb3NzOiB0aGlzLmxvc3NIaXN0b3J5W3RoaXMubG9zc0hpc3RvcnkubGVuZ3RoIC0gMV0gfHwgMCxcbiAgICAgIGJlc3RWYWxMb3NzOiB0aGlzLmJlc3RWYWxMb3NzLFxuICAgICAgYmVzdEVwb2NoOiB0aGlzLmZpbmRCZXN0RXBvY2goKSxcbiAgICAgIGFjY3VyYWN5OiAwLCAvLyBOb3QgYXBwbGljYWJsZSBmb3IgZGVjb21wb3NpdGlvblxuICAgICAgbG9zc0hpc3Rvcnk6IHRoaXMubG9zc0hpc3RvcnksXG4gICAgICB2YWxMb3NzSGlzdG9yeTogdGhpcy52YWxMb3NzSGlzdG9yeSxcbiAgICAgIGR1cmF0aW9uTXM6IERhdGUubm93KCkgLSBzdGFydFRpbWUsXG4gICAgICBlYXJseVN0b3BwZWQ6IHRoaXMucGF0aWVuY2VDb3VudGVyID49IHRoaXMuY29uZmlnLmVhcmx5U3RvcHBpbmdQYXRpZW5jZSxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIFRyYWluIG9uIHN5bnRoZXNpcyB0YXNrXG4gICAqXG4gICAqIExlYXJucyB0byBjb21iaW5lIHN1Yi1hbnN3ZXJzIGludG8gY29oZXJlbnQgZmluYWwgcmVzcG9uc2VzLlxuICAgKi9cbiAgYXN5bmMgdHJhaW5TeW50aGVzaXMoZGF0YXNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBQcm9taXNlPFRyYWluaW5nUmVzdWx0PiB7XG4gICAgY29uc3Qgc3RhcnRUaW1lID0gRGF0ZS5ub3coKTtcbiAgICB0aGlzLnJlc2V0U3RhdGUoKTtcblxuICAgIGNvbnN0IHsgdHJhaW5TZXQsIHZhbFNldCB9ID0gdGhpcy5zcGxpdERhdGFzZXQoZGF0YXNldCk7XG4gICAgY29uc3QgYmF0Y2hlcyA9IHRoaXMuY3JlYXRlQmF0Y2hlcyh0cmFpblNldCk7XG5cbiAgICBmb3IgKGxldCBlcG9jaCA9IDA7IGVwb2NoIDwgdGhpcy5jb25maWcuZXBvY2hzOyBlcG9jaCsrKSB7XG4gICAgICB0aGlzLmN1cnJlbnRFcG9jaCA9IGVwb2NoO1xuICAgICAgbGV0IGVwb2NoTG9zcyA9IDA7XG5cbiAgICAgIGZvciAoY29uc3QgYmF0Y2ggb2YgYmF0Y2hlcykge1xuICAgICAgICBjb25zdCBiYXRjaExvc3MgPSB0aGlzLnRyYWluU3ludGhlc2lzQmF0Y2goYmF0Y2gpO1xuICAgICAgICBlcG9jaExvc3MgKz0gYmF0Y2hMb3NzO1xuICAgICAgICB0aGlzLmN1cnJlbnRTdGVwKys7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGF2Z0xvc3MgPSBlcG9jaExvc3MgLyBiYXRjaGVzLmxlbmd0aDtcbiAgICAgIHRoaXMubG9zc0hpc3RvcnkucHVzaChhdmdMb3NzKTtcblxuICAgICAgLy8gVmFsaWRhdGlvblxuICAgICAgY29uc3QgdmFsTG9zcyA9IHRoaXMudmFsaWRhdGVTeW50aGVzaXModmFsU2V0KTtcbiAgICAgIHRoaXMudmFsTG9zc0hpc3RvcnkucHVzaCh2YWxMb3NzKTtcblxuICAgICAgLy8gRWFybHkgc3RvcHBpbmdcbiAgICAgIGlmICh2YWxMb3NzIDwgdGhpcy5iZXN0VmFsTG9zcykge1xuICAgICAgICB0aGlzLmJlc3RWYWxMb3NzID0gdmFsTG9zcztcbiAgICAgICAgdGhpcy5wYXRpZW5jZUNvdW50ZXIgPSAwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5wYXRpZW5jZUNvdW50ZXIrKztcbiAgICAgICAgaWYgKHRoaXMucGF0aWVuY2VDb3VudGVyID49IHRoaXMuY29uZmlnLmVhcmx5U3RvcHBpbmdQYXRpZW5jZSkge1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIHBoYXNlOiAnc3ludGhlc2lzJyxcbiAgICAgIGVwb2Noc0NvbXBsZXRlZDogdGhpcy5jdXJyZW50RXBvY2ggKyAxLFxuICAgICAgdG90YWxTdGVwczogdGhpcy5jdXJyZW50U3RlcCxcbiAgICAgIGZpbmFsTG9zczogdGhpcy5sb3NzSGlzdG9yeVt0aGlzLmxvc3NIaXN0b3J5Lmxlbmd0aCAtIDFdIHx8IDAsXG4gICAgICBiZXN0VmFsTG9zczogdGhpcy5iZXN0VmFsTG9zcyxcbiAgICAgIGJlc3RFcG9jaDogdGhpcy5maW5kQmVzdEVwb2NoKCksXG4gICAgICBhY2N1cmFjeTogMCxcbiAgICAgIGxvc3NIaXN0b3J5OiB0aGlzLmxvc3NIaXN0b3J5LFxuICAgICAgdmFsTG9zc0hpc3Rvcnk6IHRoaXMudmFsTG9zc0hpc3RvcnksXG4gICAgICBkdXJhdGlvbk1zOiBEYXRlLm5vdygpIC0gc3RhcnRUaW1lLFxuICAgICAgZWFybHlTdG9wcGVkOiB0aGlzLnBhdGllbmNlQ291bnRlciA+PSB0aGlzLmNvbmZpZy5lYXJseVN0b3BwaW5nUGF0aWVuY2UsXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb250cmFzdGl2ZSBmaW5lLXR1bmluZyBmb3IgYWdlbnQgcm91dGluZ1xuICAgKlxuICAgKiBVc2VzIHRyaXBsZXQgbG9zcyBhbmQgSW5mb05DRSB0byBpbXByb3ZlIHJvdXRpbmcgYWNjdXJhY3kuXG4gICAqL1xuICBhc3luYyB0cmFpbkNvbnRyYXN0aXZlKHBhaXJzOiBDb250cmFzdGl2ZVBhaXJbXSk6IFByb21pc2U8VHJhaW5pbmdSZXN1bHQ+IHtcbiAgICBjb25zdCBzdGFydFRpbWUgPSBEYXRlLm5vdygpO1xuICAgIHRoaXMucmVzZXRTdGF0ZSgpO1xuXG4gICAgaWYgKHBhaXJzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdObyBjb250cmFzdGl2ZSBwYWlycyBwcm92aWRlZCcpO1xuICAgIH1cblxuICAgIGNvbnN0IHsgdHJhaW5TZXQsIHZhbFNldCB9ID0gdGhpcy5zcGxpdFBhaXJzKHBhaXJzKTtcbiAgICBjb25zdCBiYXRjaGVzID0gdGhpcy5jcmVhdGVQYWlyQmF0Y2hlcyh0cmFpblNldCk7XG4gICAgbGV0IHRvdGFsQ29ycmVjdCA9IDA7XG4gICAgbGV0IHRvdGFsRXhhbXBsZXMgPSAwO1xuXG4gICAgZm9yIChsZXQgZXBvY2ggPSAwOyBlcG9jaCA8IHRoaXMuY29uZmlnLmVwb2NoczsgZXBvY2grKykge1xuICAgICAgdGhpcy5jdXJyZW50RXBvY2ggPSBlcG9jaDtcbiAgICAgIGxldCBlcG9jaExvc3MgPSAwO1xuXG4gICAgICBmb3IgKGNvbnN0IGJhdGNoIG9mIGJhdGNoZXMpIHtcbiAgICAgICAgY29uc3QgYmF0Y2hMb3NzID0gdGhpcy50cmFpbkNvbnRyYXN0aXZlQmF0Y2goYmF0Y2gpO1xuICAgICAgICBlcG9jaExvc3MgKz0gYmF0Y2hMb3NzO1xuICAgICAgICB0aGlzLmN1cnJlbnRTdGVwKys7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGF2Z0xvc3MgPSBlcG9jaExvc3MgLyBiYXRjaGVzLmxlbmd0aDtcbiAgICAgIHRoaXMubG9zc0hpc3RvcnkucHVzaChhdmdMb3NzKTtcblxuICAgICAgLy8gVmFsaWRhdGlvblxuICAgICAgY29uc3QgeyBsb3NzOiB2YWxMb3NzLCBjb3JyZWN0LCB0b3RhbCB9ID0gdGhpcy52YWxpZGF0ZUNvbnRyYXN0aXZlKHZhbFNldCk7XG4gICAgICB0aGlzLnZhbExvc3NIaXN0b3J5LnB1c2godmFsTG9zcyk7XG4gICAgICB0b3RhbENvcnJlY3QgPSBjb3JyZWN0O1xuICAgICAgdG90YWxFeGFtcGxlcyA9IHRvdGFsO1xuXG4gICAgICAvLyBFYXJseSBzdG9wcGluZ1xuICAgICAgaWYgKHZhbExvc3MgPCB0aGlzLmJlc3RWYWxMb3NzKSB7XG4gICAgICAgIHRoaXMuYmVzdFZhbExvc3MgPSB2YWxMb3NzO1xuICAgICAgICB0aGlzLnBhdGllbmNlQ291bnRlciA9IDA7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLnBhdGllbmNlQ291bnRlcisrO1xuICAgICAgICBpZiAodGhpcy5wYXRpZW5jZUNvdW50ZXIgPj0gdGhpcy5jb25maWcuZWFybHlTdG9wcGluZ1BhdGllbmNlKSB7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgcGhhc2U6ICdjb250cmFzdGl2ZScsXG4gICAgICBlcG9jaHNDb21wbGV0ZWQ6IHRoaXMuY3VycmVudEVwb2NoICsgMSxcbiAgICAgIHRvdGFsU3RlcHM6IHRoaXMuY3VycmVudFN0ZXAsXG4gICAgICBmaW5hbExvc3M6IHRoaXMubG9zc0hpc3RvcnlbdGhpcy5sb3NzSGlzdG9yeS5sZW5ndGggLSAxXSB8fCAwLFxuICAgICAgYmVzdFZhbExvc3M6IHRoaXMuYmVzdFZhbExvc3MsXG4gICAgICBiZXN0RXBvY2g6IHRoaXMuZmluZEJlc3RFcG9jaCgpLFxuICAgICAgYWNjdXJhY3k6IHRvdGFsRXhhbXBsZXMgPiAwID8gdG90YWxDb3JyZWN0IC8gdG90YWxFeGFtcGxlcyA6IDAsXG4gICAgICBsb3NzSGlzdG9yeTogdGhpcy5sb3NzSGlzdG9yeSxcbiAgICAgIHZhbExvc3NIaXN0b3J5OiB0aGlzLnZhbExvc3NIaXN0b3J5LFxuICAgICAgZHVyYXRpb25NczogRGF0ZS5ub3coKSAtIHN0YXJ0VGltZSxcbiAgICAgIGVhcmx5U3RvcHBlZDogdGhpcy5wYXRpZW5jZUNvdW50ZXIgPj0gdGhpcy5jb25maWcuZWFybHlTdG9wcGluZ1BhdGllbmNlLFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogRXZhbHVhdGUgdHJhaW5lZCBtb2RlbCBvbiB0ZXN0IHNldFxuICAgKi9cbiAgYXN5bmMgZXZhbHVhdGUodGVzdFNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBQcm9taXNlPEV2YWx1YXRpb25SZXN1bHQ+IHtcbiAgICBjb25zdCBwZXJBZ2VudEFjY3VyYWN5OiBSZWNvcmQ8c3RyaW5nLCB7IGNvcnJlY3Q6IG51bWJlcjsgdG90YWw6IG51bWJlciB9PiA9IHt9O1xuXG4gICAgbGV0IGRlY29tcG9zaXRpb25Db3JyZWN0ID0gMDtcbiAgICBsZXQgc3ludGhlc2lzUXVhbGl0eVN1bSA9IDA7XG4gICAgbGV0IHJvdXRpbmdDb3JyZWN0ID0gMDtcbiAgICBsZXQgaGFyZE5lZ2F0aXZlQ29ycmVjdCA9IDA7XG4gICAgbGV0IGhhcmROZWdhdGl2ZVRvdGFsID0gMDtcbiAgICBsZXQgdG90YWxMYXRlbmN5ID0gMDtcblxuICAgIGZvciAoY29uc3QgZXhhbXBsZSBvZiB0ZXN0U2V0KSB7XG4gICAgICAvLyBEZWNvbXBvc2l0aW9uIGV2YWx1YXRpb25cbiAgICAgIGlmIChleGFtcGxlLmRlY29tcG9zaXRpb24uc3VjY2VzcyAmJiBleGFtcGxlLmRlY29tcG9zaXRpb24uc3ViUXVlcmllcy5sZW5ndGggPiAwKSB7XG4gICAgICAgIGRlY29tcG9zaXRpb25Db3JyZWN0Kys7XG4gICAgICB9XG5cbiAgICAgIC8vIFN5bnRoZXNpcyBxdWFsaXR5XG4gICAgICBzeW50aGVzaXNRdWFsaXR5U3VtICs9IGV4YW1wbGUucXVhbGl0eVNjb3JlO1xuXG4gICAgICAvLyBSb3V0aW5nIGV2YWx1YXRpb25cbiAgICAgIGZvciAoY29uc3Qgc3ViUXVlcnkgb2YgZXhhbXBsZS5kZWNvbXBvc2l0aW9uLnN1YlF1ZXJpZXMpIHtcbiAgICAgICAgaWYgKHN1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnQpIHtcbiAgICAgICAgICBjb25zdCBwcmVkaWN0ZWQgPSB0aGlzLnByZWRpY3RBZ2VudChzdWJRdWVyeS5xdWVyeSk7XG4gICAgICAgICAgY29uc3QgY29ycmVjdCA9IHByZWRpY3RlZCA9PT0gc3ViUXVlcnkucmVjb21tZW5kZWRBZ2VudDtcblxuICAgICAgICAgIGlmIChjb3JyZWN0KSB7XG4gICAgICAgICAgICByb3V0aW5nQ29ycmVjdCsrO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIC8vIFRyYWNrIHBlci1hZ2VudCBhY2N1cmFjeVxuICAgICAgICAgIGlmICghcGVyQWdlbnRBY2N1cmFjeVtzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50XSkge1xuICAgICAgICAgICAgcGVyQWdlbnRBY2N1cmFjeVtzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50XSA9IHsgY29ycmVjdDogMCwgdG90YWw6IDAgfTtcbiAgICAgICAgICB9XG4gICAgICAgICAgcGVyQWdlbnRBY2N1cmFjeVtzdWJRdWVyeS5yZWNvbW1lbmRlZEFnZW50XS50b3RhbCsrO1xuICAgICAgICAgIGlmIChjb3JyZWN0KSB7XG4gICAgICAgICAgICBwZXJBZ2VudEFjY3VyYWN5W3N1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnRdLmNvcnJlY3QrKztcbiAgICAgICAgICB9XG5cbiAgICAgICAgICAvLyBDaGVjayBoYXJkIG5lZ2F0aXZlc1xuICAgICAgICAgIGlmICh0aGlzLmlzSGFyZE5lZ2F0aXZlKHN1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnQsIHByZWRpY3RlZCkpIHtcbiAgICAgICAgICAgIGhhcmROZWdhdGl2ZVRvdGFsKys7XG4gICAgICAgICAgICBpZiAoY29ycmVjdCkge1xuICAgICAgICAgICAgICBoYXJkTmVnYXRpdmVDb3JyZWN0Kys7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHRvdGFsTGF0ZW5jeSArPSBleGFtcGxlLnRyYWplY3RvcnkudG90YWxMYXRlbmN5TXM7XG4gICAgfVxuXG4gICAgY29uc3QgdG90YWxSb3V0aW5nRXhhbXBsZXMgPSB0ZXN0U2V0LnJlZHVjZShcbiAgICAgIChzdW0sIGV4KSA9PiBzdW0gKyBleC5kZWNvbXBvc2l0aW9uLnN1YlF1ZXJpZXMuZmlsdGVyKChzcSkgPT4gc3EucmVjb21tZW5kZWRBZ2VudCkubGVuZ3RoLFxuICAgICAgMFxuICAgICk7XG5cbiAgICBjb25zdCBwZXJBZ2VudFJlc3VsdDogUmVjb3JkPHN0cmluZywgbnVtYmVyPiA9IHt9O1xuICAgIGZvciAoY29uc3QgW2FnZW50LCBzdGF0c10gb2YgT2JqZWN0LmVudHJpZXMocGVyQWdlbnRBY2N1cmFjeSkpIHtcbiAgICAgIHBlckFnZW50UmVzdWx0W2FnZW50XSA9IHN0YXRzLnRvdGFsID4gMCA/IHN0YXRzLmNvcnJlY3QgLyBzdGF0cy50b3RhbCA6IDA7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGRlY29tcG9zaXRpb25BY2N1cmFjeTogdGVzdFNldC5sZW5ndGggPiAwID8gZGVjb21wb3NpdGlvbkNvcnJlY3QgLyB0ZXN0U2V0Lmxlbmd0aCA6IDAsXG4gICAgICBzeW50aGVzaXNRdWFsaXR5OiB0ZXN0U2V0Lmxlbmd0aCA+IDAgPyBzeW50aGVzaXNRdWFsaXR5U3VtIC8gdGVzdFNldC5sZW5ndGggOiAwLFxuICAgICAgcm91dGluZ0FjY3VyYWN5OiB0b3RhbFJvdXRpbmdFeGFtcGxlcyA+IDAgPyByb3V0aW5nQ29ycmVjdCAvIHRvdGFsUm91dGluZ0V4YW1wbGVzIDogMCxcbiAgICAgIGhhcmROZWdhdGl2ZUFjY3VyYWN5OiBoYXJkTmVnYXRpdmVUb3RhbCA+IDAgPyBoYXJkTmVnYXRpdmVDb3JyZWN0IC8gaGFyZE5lZ2F0aXZlVG90YWwgOiAwLFxuICAgICAgYXZnTGF0ZW5jeU1zOiB0ZXN0U2V0Lmxlbmd0aCA+IDAgPyB0b3RhbExhdGVuY3kgLyB0ZXN0U2V0Lmxlbmd0aCA6IDAsXG4gICAgICB0b3RhbEV4YW1wbGVzOiB0ZXN0U2V0Lmxlbmd0aCxcbiAgICAgIHBlckFnZW50QWNjdXJhY3k6IHBlckFnZW50UmVzdWx0LFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogR2VuZXJhdGUgY29udHJhc3RpdmUgcGFpcnMgZnJvbSBkYXRhc2V0XG4gICAqL1xuICBnZW5lcmF0ZUNvbnRyYXN0aXZlUGFpcnMoXG4gICAgZGF0YXNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10sXG4gICAgaGFyZE5lZ2F0aXZlUmF0aW8gPSAwLjNcbiAgKTogQ29udHJhc3RpdmVQYWlyW10ge1xuICAgIGNvbnN0IHBhaXJzOiBDb250cmFzdGl2ZVBhaXJbXSA9IFtdO1xuICAgIGNvbnN0IGFnZW50cyA9IE9iamVjdC5rZXlzKEFHRU5UX0RFRklOSVRJT05TKTtcblxuICAgIGZvciAoY29uc3QgZXhhbXBsZSBvZiBkYXRhc2V0KSB7XG4gICAgICBmb3IgKGNvbnN0IHN1YlF1ZXJ5IG9mIGV4YW1wbGUuZGVjb21wb3NpdGlvbi5zdWJRdWVyaWVzKSB7XG4gICAgICAgIGlmICghc3ViUXVlcnkucmVjb21tZW5kZWRBZ2VudCkgY29udGludWU7XG5cbiAgICAgICAgY29uc3QgcG9zaXRpdmVBZ2VudCA9IHN1YlF1ZXJ5LnJlY29tbWVuZGVkQWdlbnQ7XG5cbiAgICAgICAgZm9yIChjb25zdCBuZWdhdGl2ZUFnZW50IG9mIGFnZW50cykge1xuICAgICAgICAgIGlmIChuZWdhdGl2ZUFnZW50ID09PSBwb3NpdGl2ZUFnZW50KSBjb250aW51ZTtcblxuICAgICAgICAgIGNvbnN0IGlzSGFyZCA9IHRoaXMuaXNIYXJkTmVnYXRpdmUocG9zaXRpdmVBZ2VudCwgbmVnYXRpdmVBZ2VudCk7XG5cbiAgICAgICAgICAvLyBBcHBseSBoYXJkIG5lZ2F0aXZlIHJhdGlvXG4gICAgICAgICAgY29uc3QgaW5jbHVkZSA9IGlzSGFyZFxuICAgICAgICAgICAgPyBNYXRoLnJhbmRvbSgpIDwgaGFyZE5lZ2F0aXZlUmF0aW9cbiAgICAgICAgICAgIDogTWF0aC5yYW5kb20oKSA8IDEgLSBoYXJkTmVnYXRpdmVSYXRpbztcblxuICAgICAgICAgIGlmIChpbmNsdWRlKSB7XG4gICAgICAgICAgICBwYWlycy5wdXNoKHtcbiAgICAgICAgICAgICAgYW5jaG9yOiBzdWJRdWVyeS5xdWVyeSxcbiAgICAgICAgICAgICAgYW5jaG9yRW1iZWRkaW5nOiBleGFtcGxlLnF1ZXJ5RW1iZWRkaW5nLFxuICAgICAgICAgICAgICBwb3NpdGl2ZUFnZW50LFxuICAgICAgICAgICAgICBuZWdhdGl2ZUFnZW50LFxuICAgICAgICAgICAgICBpc0hhcmROZWdhdGl2ZTogaXNIYXJkLFxuICAgICAgICAgICAgICBxdWFsaXR5OiBleGFtcGxlLnF1YWxpdHlTY29yZSxcbiAgICAgICAgICAgICAgc291cmNlSWQ6IGV4YW1wbGUuaWQsXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gcGFpcnM7XG4gIH1cblxuICAvLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICAvLyBQcml2YXRlIE1ldGhvZHNcbiAgLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuICBwcml2YXRlIHJlc2V0U3RhdGUoKTogdm9pZCB7XG4gICAgdGhpcy5jdXJyZW50RXBvY2ggPSAwO1xuICAgIHRoaXMuY3VycmVudFN0ZXAgPSAwO1xuICAgIHRoaXMuYmVzdFZhbExvc3MgPSBJbmZpbml0eTtcbiAgICB0aGlzLnBhdGllbmNlQ291bnRlciA9IDA7XG4gICAgdGhpcy5sb3NzSGlzdG9yeSA9IFtdO1xuICAgIHRoaXMudmFsTG9zc0hpc3RvcnkgPSBbXTtcbiAgfVxuXG4gIHByaXZhdGUgc3BsaXREYXRhc2V0KFxuICAgIGRhdGFzZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdXG4gICk6IHsgdHJhaW5TZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdOyB2YWxTZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdIH0ge1xuICAgIGNvbnN0IHZhbFNpemUgPSBNYXRoLmZsb29yKGRhdGFzZXQubGVuZ3RoICogdGhpcy5jb25maWcudmFsaWRhdGlvblNwbGl0KTtcbiAgICBjb25zdCBzaHVmZmxlZCA9IHRoaXMuc2h1ZmZsZShbLi4uZGF0YXNldF0pO1xuICAgIHJldHVybiB7XG4gICAgICB0cmFpblNldDogc2h1ZmZsZWQuc2xpY2UodmFsU2l6ZSksXG4gICAgICB2YWxTZXQ6IHNodWZmbGVkLnNsaWNlKDAsIHZhbFNpemUpLFxuICAgIH07XG4gIH1cblxuICBwcml2YXRlIHNwbGl0UGFpcnMoXG4gICAgcGFpcnM6IENvbnRyYXN0aXZlUGFpcltdXG4gICk6IHsgdHJhaW5TZXQ6IENvbnRyYXN0aXZlUGFpcltdOyB2YWxTZXQ6IENvbnRyYXN0aXZlUGFpcltdIH0ge1xuICAgIGNvbnN0IHZhbFNpemUgPSBNYXRoLmZsb29yKHBhaXJzLmxlbmd0aCAqIHRoaXMuY29uZmlnLnZhbGlkYXRpb25TcGxpdCk7XG4gICAgY29uc3Qgc2h1ZmZsZWQgPSB0aGlzLnNodWZmbGUoWy4uLnBhaXJzXSk7XG4gICAgcmV0dXJuIHtcbiAgICAgIHRyYWluU2V0OiBzaHVmZmxlZC5zbGljZSh2YWxTaXplKSxcbiAgICAgIHZhbFNldDogc2h1ZmZsZWQuc2xpY2UoMCwgdmFsU2l6ZSksXG4gICAgfTtcbiAgfVxuXG4gIHByaXZhdGUgY3JlYXRlQmF0Y2hlcyhkYXRhc2V0OiBSbG1UcmFpbmluZ0V4YW1wbGVbXSk6IFJsbVRyYWluaW5nRXhhbXBsZVtdW10ge1xuICAgIGNvbnN0IGJhdGNoZXM6IFJsbVRyYWluaW5nRXhhbXBsZVtdW10gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGRhdGFzZXQubGVuZ3RoOyBpICs9IHRoaXMuY29uZmlnLmJhdGNoU2l6ZSkge1xuICAgICAgYmF0Y2hlcy5wdXNoKGRhdGFzZXQuc2xpY2UoaSwgaSArIHRoaXMuY29uZmlnLmJhdGNoU2l6ZSkpO1xuICAgIH1cbiAgICByZXR1cm4gYmF0Y2hlcztcbiAgfVxuXG4gIHByaXZhdGUgY3JlYXRlUGFpckJhdGNoZXMocGFpcnM6IENvbnRyYXN0aXZlUGFpcltdKTogQ29udHJhc3RpdmVQYWlyW11bXSB7XG4gICAgY29uc3QgYmF0Y2hlczogQ29udHJhc3RpdmVQYWlyW11bXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcGFpcnMubGVuZ3RoOyBpICs9IHRoaXMuY29uZmlnLmJhdGNoU2l6ZSkge1xuICAgICAgYmF0Y2hlcy5wdXNoKHBhaXJzLnNsaWNlKGksIGkgKyB0aGlzLmNvbmZpZy5iYXRjaFNpemUpKTtcbiAgICB9XG4gICAgcmV0dXJuIGJhdGNoZXM7XG4gIH1cblxuICBwcml2YXRlIHNodWZmbGU8VD4oYXJyYXk6IFRbXSk6IFRbXSB7XG4gICAgLy8gRmlzaGVyLVlhdGVzIHNodWZmbGVcbiAgICBmb3IgKGxldCBpID0gYXJyYXkubGVuZ3RoIC0gMTsgaSA+IDA7IGktLSkge1xuICAgICAgY29uc3QgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIChpICsgMSkpO1xuICAgICAgW2FycmF5W2ldLCBhcnJheVtqXV0gPSBbYXJyYXlbal0sIGFycmF5W2ldXTtcbiAgICB9XG4gICAgcmV0dXJuIGFycmF5O1xuICB9XG5cbiAgcHJpdmF0ZSB0cmFpbkRlY29tcG9zaXRpb25CYXRjaChiYXRjaDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBudW1iZXIge1xuICAgIGxldCBiYXRjaExvc3MgPSAwO1xuXG4gICAgZm9yIChjb25zdCBleGFtcGxlIG9mIGJhdGNoKSB7XG4gICAgICAvLyBEZWNvbXBvc2l0aW9uIHF1YWxpdHkgbG9zc1xuICAgICAgY29uc3QgcXVhbGl0eUxvc3MgPSAxIC0gKGV4YW1wbGUuZGVjb21wb3NpdGlvbi5zdWNjZXNzID8gZXhhbXBsZS5xdWFsaXR5U2NvcmUgOiAwKTtcblxuICAgICAgLy8gRGVwdGggYXBwcm9wcmlhdGVuZXNzIChwZW5hbGl6ZSB0b28gc2hhbGxvdyBvciB0b28gZGVlcClcbiAgICAgIGNvbnN0IGRlcHRoID0gZXhhbXBsZS5kZWNvbXBvc2l0aW9uLnN1YlF1ZXJpZXMubGVuZ3RoO1xuICAgICAgY29uc3QgaWRlYWxEZXB0aCA9IDM7XG4gICAgICBjb25zdCBkZXB0aExvc3MgPSBNYXRoLmFicyhkZXB0aCAtIGlkZWFsRGVwdGgpIC8gaWRlYWxEZXB0aDtcblxuICAgICAgLy8gQ29tcGxleGl0eSBiYWxhbmNlIGxvc3NcbiAgICAgIGNvbnN0IGNvbXBsZXhpdHlMb3NzID0gTWF0aC5hYnMoZXhhbXBsZS5kZWNvbXBvc2l0aW9uLnRvdGFsQ29tcGxleGl0eSAtIDEpIC8gMztcblxuICAgICAgY29uc3QgbG9zcyA9XG4gICAgICAgIHF1YWxpdHlMb3NzICogdGhpcy5jb25maWcuZGVjb21wb3NpdGlvbldlaWdodCAqIDAuNiArXG4gICAgICAgIGRlcHRoTG9zcyAqIDAuMiArXG4gICAgICAgIGNvbXBsZXhpdHlMb3NzICogMC4yO1xuXG4gICAgICBiYXRjaExvc3MgKz0gbG9zcztcbiAgICB9XG5cbiAgICByZXR1cm4gYmF0Y2hMb3NzIC8gYmF0Y2gubGVuZ3RoO1xuICB9XG5cbiAgcHJpdmF0ZSB0cmFpblN5bnRoZXNpc0JhdGNoKGJhdGNoOiBSbG1UcmFpbmluZ0V4YW1wbGVbXSk6IG51bWJlciB7XG4gICAgbGV0IGJhdGNoTG9zcyA9IDA7XG5cbiAgICBmb3IgKGNvbnN0IGV4YW1wbGUgb2YgYmF0Y2gpIHtcbiAgICAgIC8vIFN1Yi1hbnN3ZXIgcXVhbGl0eVxuICAgICAgY29uc3Qgc3ViQW5zd2VyUXVhbGl0eSA9XG4gICAgICAgIGV4YW1wbGUuc3ViQW5zd2Vycy5sZW5ndGggPiAwXG4gICAgICAgICAgPyBleGFtcGxlLnN1YkFuc3dlcnMucmVkdWNlKChzdW0sIGEpID0+IHN1bSArIGEucXVhbGl0eSwgMCkgLyBleGFtcGxlLnN1YkFuc3dlcnMubGVuZ3RoXG4gICAgICAgICAgOiAwO1xuXG4gICAgICAvLyBGaW5hbCBhbnN3ZXIgcXVhbGl0eVxuICAgICAgY29uc3QgZmluYWxRdWFsaXR5ID0gZXhhbXBsZS5xdWFsaXR5U2NvcmU7XG5cbiAgICAgIC8vIENvaGVyZW5jZSBib251cyAoZmluYWwgc2hvdWxkIGJlIGJldHRlciB0aGFuIHBhcnRzIGF2ZXJhZ2UpXG4gICAgICBjb25zdCBjb2hlcmVuY2VCb251cyA9IE1hdGgubWF4KDAsIGZpbmFsUXVhbGl0eSAtIHN1YkFuc3dlclF1YWxpdHkpICogMC41O1xuXG4gICAgICBjb25zdCBsb3NzID0gKDEgLSAoc3ViQW5zd2VyUXVhbGl0eSAqIDAuNCArIGZpbmFsUXVhbGl0eSAqIDAuNCArIGNvaGVyZW5jZUJvbnVzICogMC4yKSk7XG5cbiAgICAgIGJhdGNoTG9zcyArPSBsb3NzICogdGhpcy5jb25maWcuc3ludGhlc2lzV2VpZ2h0O1xuICAgIH1cblxuICAgIHJldHVybiBiYXRjaExvc3MgLyBiYXRjaC5sZW5ndGg7XG4gIH1cblxuICBwcml2YXRlIHRyYWluQ29udHJhc3RpdmVCYXRjaChiYXRjaDogQ29udHJhc3RpdmVQYWlyW10pOiBudW1iZXIge1xuICAgIGxldCBiYXRjaExvc3MgPSAwO1xuXG4gICAgZm9yIChjb25zdCBwYWlyIG9mIGJhdGNoKSB7XG4gICAgICAvLyBUcmlwbGV0IGxvc3NcbiAgICAgIGNvbnN0IHRyaXBsZXRMb3NzID0gdGhpcy5jb21wdXRlVHJpcGxldExvc3MocGFpcik7XG5cbiAgICAgIC8vIEluZm9OQ0UgbG9zc1xuICAgICAgY29uc3QgaW5mb25jZUxvc3MgPSB0aGlzLmNvbXB1dGVJbmZvTkNFTG9zcyhwYWlyKTtcblxuICAgICAgYmF0Y2hMb3NzICs9ICh0cmlwbGV0TG9zcyAqIDAuNSArIGluZm9uY2VMb3NzICogMC41KSAqIHRoaXMuY29uZmlnLnJvdXRpbmdXZWlnaHQ7XG4gICAgfVxuXG4gICAgcmV0dXJuIGJhdGNoTG9zcyAvIGJhdGNoLmxlbmd0aDtcbiAgfVxuXG4gIHByaXZhdGUgdmFsaWRhdGVEZWNvbXBvc2l0aW9uKHZhbFNldDogUmxtVHJhaW5pbmdFeGFtcGxlW10pOiBudW1iZXIge1xuICAgIGlmICh2YWxTZXQubGVuZ3RoID09PSAwKSByZXR1cm4gMDtcblxuICAgIGxldCB0b3RhbExvc3MgPSAwO1xuICAgIGZvciAoY29uc3QgZXhhbXBsZSBvZiB2YWxTZXQpIHtcbiAgICAgIHRvdGFsTG9zcyArPSAxIC0gZXhhbXBsZS5xdWFsaXR5U2NvcmU7XG4gICAgfVxuICAgIHJldHVybiB0b3RhbExvc3MgLyB2YWxTZXQubGVuZ3RoO1xuICB9XG5cbiAgcHJpdmF0ZSB2YWxpZGF0ZVN5bnRoZXNpcyh2YWxTZXQ6IFJsbVRyYWluaW5nRXhhbXBsZVtdKTogbnVtYmVyIHtcbiAgICBpZiAodmFsU2V0Lmxlbmd0aCA9PT0gMCkgcmV0dXJuIDA7XG5cbiAgICBsZXQgdG90YWxMb3NzID0gMDtcbiAgICBmb3IgKGNvbnN0IGV4YW1wbGUgb2YgdmFsU2V0KSB7XG4gICAgICB0b3RhbExvc3MgKz0gMSAtIGV4YW1wbGUucXVhbGl0eVNjb3JlO1xuICAgIH1cbiAgICByZXR1cm4gdG90YWxMb3NzIC8gdmFsU2V0Lmxlbmd0aDtcbiAgfVxuXG4gIHByaXZhdGUgdmFsaWRhdGVDb250cmFzdGl2ZShcbiAgICB2YWxTZXQ6IENvbnRyYXN0aXZlUGFpcltdXG4gICk6IHsgbG9zczogbnVtYmVyOyBjb3JyZWN0OiBudW1iZXI7IHRvdGFsOiBudW1iZXIgfSB7XG4gICAgaWYgKHZhbFNldC5sZW5ndGggPT09IDApIHJldHVybiB7IGxvc3M6IDAsIGNvcnJlY3Q6IDAsIHRvdGFsOiAwIH07XG5cbiAgICBsZXQgdG90YWxMb3NzID0gMDtcbiAgICBsZXQgY29ycmVjdCA9IDA7XG5cbiAgICBmb3IgKGNvbnN0IHBhaXIgb2YgdmFsU2V0KSB7XG4gICAgICBjb25zdCB0cmlwbGV0TG9zcyA9IHRoaXMuY29tcHV0ZVRyaXBsZXRMb3NzKHBhaXIpO1xuICAgICAgY29uc3QgaW5mb25jZUxvc3MgPSB0aGlzLmNvbXB1dGVJbmZvTkNFTG9zcyhwYWlyKTtcbiAgICAgIHRvdGFsTG9zcyArPSB0cmlwbGV0TG9zcyAqIDAuNSArIGluZm9uY2VMb3NzICogMC41O1xuXG4gICAgICAvLyBDaGVjayByb3V0aW5nIGNvcnJlY3RuZXNzXG4gICAgICBjb25zdCBwb3NEaXN0ID0gdGhpcy5hZ2VudERpc3RhbmNlKHBhaXIuYW5jaG9yLCBwYWlyLnBvc2l0aXZlQWdlbnQpO1xuICAgICAgY29uc3QgbmVnRGlzdCA9IHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5uZWdhdGl2ZUFnZW50KTtcbiAgICAgIGlmIChwb3NEaXN0IDwgbmVnRGlzdCkge1xuICAgICAgICBjb3JyZWN0Kys7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGxvc3M6IHRvdGFsTG9zcyAvIHZhbFNldC5sZW5ndGgsXG4gICAgICBjb3JyZWN0LFxuICAgICAgdG90YWw6IHZhbFNldC5sZW5ndGgsXG4gICAgfTtcbiAgfVxuXG4gIHByaXZhdGUgY29tcHV0ZVRyaXBsZXRMb3NzKHBhaXI6IENvbnRyYXN0aXZlUGFpcik6IG51bWJlciB7XG4gICAgY29uc3QgcG9zRGlzdCA9IHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5wb3NpdGl2ZUFnZW50KTtcbiAgICBjb25zdCBuZWdEaXN0ID0gdGhpcy5hZ2VudERpc3RhbmNlKHBhaXIuYW5jaG9yLCBwYWlyLm5lZ2F0aXZlQWdlbnQpO1xuICAgIHJldHVybiBNYXRoLm1heCgwLCB0aGlzLmNvbmZpZy5jb250cmFzdGl2ZU1hcmdpbiArIHBvc0Rpc3QgLSBuZWdEaXN0KTtcbiAgfVxuXG4gIHByaXZhdGUgY29tcHV0ZUluZm9OQ0VMb3NzKHBhaXI6IENvbnRyYXN0aXZlUGFpcik6IG51bWJlciB7XG4gICAgY29uc3QgcG9zU2ltID0gMSAtIHRoaXMuYWdlbnREaXN0YW5jZShwYWlyLmFuY2hvciwgcGFpci5wb3NpdGl2ZUFnZW50KTtcbiAgICBjb25zdCBuZWdTaW0gPSAxIC0gdGhpcy5hZ2VudERpc3RhbmNlKHBhaXIuYW5jaG9yLCBwYWlyLm5lZ2F0aXZlQWdlbnQpO1xuXG4gICAgY29uc3QgdGVtcCA9IHRoaXMuY29uZmlnLmluZm9uY2VUZW1wZXJhdHVyZTtcbiAgICBjb25zdCBwb3NFeHAgPSBNYXRoLmV4cChwb3NTaW0gLyB0ZW1wKTtcbiAgICBjb25zdCBuZWdFeHAgPSBNYXRoLmV4cChuZWdTaW0gLyB0ZW1wKTtcblxuICAgIHJldHVybiAtTWF0aC5sb2cocG9zRXhwIC8gKHBvc0V4cCArIG5lZ0V4cCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBhZ2VudERpc3RhbmNlKHF1ZXJ5OiBzdHJpbmcsIGFnZW50OiBzdHJpbmcpOiBudW1iZXIge1xuICAgIGNvbnN0IHF1ZXJ5TG93ZXIgPSBxdWVyeS50b0xvd2VyQ2FzZSgpO1xuICAgIGNvbnN0IGFnZW50RGVmID0gQUdFTlRfREVGSU5JVElPTlNbYWdlbnRdO1xuXG4gICAgaWYgKCFhZ2VudERlZikgcmV0dXJuIDEuMDtcblxuICAgIGNvbnN0IG1hdGNoZXMgPSBhZ2VudERlZi5rZXl3b3Jkcy5maWx0ZXIoKGt3KSA9PiBxdWVyeUxvd2VyLmluY2x1ZGVzKGt3KSkubGVuZ3RoO1xuICAgIHJldHVybiAxLjAgLSBNYXRoLm1pbigxLjAsIG1hdGNoZXMgLyBhZ2VudERlZi5rZXl3b3Jkcy5sZW5ndGgpO1xuICB9XG5cbiAgcHJpdmF0ZSBwcmVkaWN0QWdlbnQocXVlcnk6IHN0cmluZyk6IHN0cmluZyB7XG4gICAgbGV0IGJlc3RBZ2VudCA9ICdjb2Rlcic7XG4gICAgbGV0IGJlc3RTY29yZSA9IDA7XG5cbiAgICBmb3IgKGNvbnN0IFthZ2VudCwgZGVmXSBvZiBPYmplY3QuZW50cmllcyhBR0VOVF9ERUZJTklUSU9OUykpIHtcbiAgICAgIGNvbnN0IHF1ZXJ5TG93ZXIgPSBxdWVyeS50b0xvd2VyQ2FzZSgpO1xuICAgICAgY29uc3QgbWF0Y2hlcyA9IGRlZi5rZXl3b3Jkcy5maWx0ZXIoKGt3KSA9PiBxdWVyeUxvd2VyLmluY2x1ZGVzKGt3KSkubGVuZ3RoO1xuICAgICAgY29uc3Qgc2NvcmUgPSBtYXRjaGVzIC8gZGVmLmtleXdvcmRzLmxlbmd0aDtcblxuICAgICAgaWYgKHNjb3JlID4gYmVzdFNjb3JlKSB7XG4gICAgICAgIGJlc3RTY29yZSA9IHNjb3JlO1xuICAgICAgICBiZXN0QWdlbnQgPSBhZ2VudDtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYmVzdEFnZW50O1xuICB9XG5cbiAgcHJpdmF0ZSBpc0hhcmROZWdhdGl2ZShhZ2VudDE6IHN0cmluZywgYWdlbnQyOiBzdHJpbmcpOiBib29sZWFuIHtcbiAgICByZXR1cm4gSEFSRF9ORUdBVElWRV9QQUlSUy5zb21lKFxuICAgICAgKFthLCBiXSkgPT4gKGFnZW50MSA9PT0gYSAmJiBhZ2VudDIgPT09IGIpIHx8IChhZ2VudDEgPT09IGIgJiYgYWdlbnQyID09PSBhKVxuICAgICk7XG4gIH1cblxuICBwcml2YXRlIGZpbmRCZXN0RXBvY2goKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy52YWxMb3NzSGlzdG9yeS5sZW5ndGggPT09IDApIHJldHVybiAwO1xuXG4gICAgbGV0IGJlc3RJZHggPSAwO1xuICAgIGxldCBiZXN0TG9zcyA9IHRoaXMudmFsTG9zc0hpc3RvcnlbMF07XG5cbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IHRoaXMudmFsTG9zc0hpc3RvcnkubGVuZ3RoOyBpKyspIHtcbiAgICAgIGlmICh0aGlzLnZhbExvc3NIaXN0b3J5W2ldIDwgYmVzdExvc3MpIHtcbiAgICAgICAgYmVzdExvc3MgPSB0aGlzLnZhbExvc3NIaXN0b3J5W2ldO1xuICAgICAgICBiZXN0SWR4ID0gaTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYmVzdElkeDtcbiAgfVxufVxuXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuLy8gRmFjdG9yeSBGdW5jdGlvbnNcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8qKlxuICogQ3JlYXRlIGFuIFJMTSB0cmFpbmVyIHdpdGggZGVmYXVsdCBjb25maWd1cmF0aW9uXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVSbG1UcmFpbmVyKGNvbmZpZz86IFBhcnRpYWw8UmxtVHJhaW5pbmdDb25maWc+KTogUmxtVHJhaW5lciB7XG4gIHJldHVybiBuZXcgUmxtVHJhaW5lcihjb25maWcpO1xufVxuXG4vKipcbiAqIENyZWF0ZSBhbiBlbXB0eSBSTE0gdHJhaW5pbmcgZXhhbXBsZVxuICovXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRW1wdHlFeGFtcGxlKHF1ZXJ5OiBzdHJpbmcpOiBSbG1UcmFpbmluZ0V4YW1wbGUge1xuICByZXR1cm4ge1xuICAgIGlkOiBjcnlwdG8ucmFuZG9tVVVJRCA/IGNyeXB0by5yYW5kb21VVUlEKCkgOiBgJHtEYXRlLm5vdygpfS0ke01hdGgucmFuZG9tKCkudG9TdHJpbmcoMzYpfWAsXG4gICAgcXVlcnksXG4gICAgZGVjb21wb3NpdGlvbjoge1xuICAgICAgc3ViUXVlcmllczogW10sXG4gICAgICBzdHJhdGVneTogJ25vbmUnLFxuICAgICAgcmF0aW9uYWxlOiAnJyxcbiAgICAgIHRvdGFsQ29tcGxleGl0eTogMCxcbiAgICAgIHN1Y2Nlc3M6IGZhbHNlLFxuICAgIH0sXG4gICAgc3ViQW5zd2VyczogW10sXG4gICAgZmluYWxBbnN3ZXI6ICcnLFxuICAgIHF1YWxpdHlTY29yZTogMCxcbiAgICB0cmFqZWN0b3J5OiB7XG4gICAgICB0b3RhbExhdGVuY3lNczogMCxcbiAgICAgIHJldHJpZXM6IDAsXG4gICAgICBtYXhQYXJhbGxlbGlzbTogMSxcbiAgICAgIG1vZGVsc1VzZWQ6IFtdLFxuICAgICAgYWdlbnRzSW52b2tlZDogW10sXG4gICAgICB0b29sc1VzZWQ6IFtdLFxuICAgICAgYXR0cmlidXRlczoge30sXG4gICAgfSxcbiAgICBzdWNjZXNzOiBmYWxzZSxcbiAgICBsZXNzb25zOiBbXSxcbiAgICBzb3VyY2U6ICdtYW51YWwnLFxuICB9O1xufVxuXG4vKipcbiAqIENyZWF0ZSBhIHN1Yi1xdWVyeVxuICovXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3ViUXVlcnkoXG4gIGlkOiBudW1iZXIsXG4gIHF1ZXJ5OiBzdHJpbmcsXG4gIG9wdGlvbnM6IFBhcnRpYWw8U3ViUXVlcnk+ID0ge31cbik6IFN1YlF1ZXJ5IHtcbiAgcmV0dXJuIHtcbiAgICBpZCxcbiAgICBxdWVyeSxcbiAgICBleHBlY3RlZFR5cGU6ICd0ZXh0JyxcbiAgICBkZXBlbmRlbmNpZXM6IFtdLFxuICAgIGNvbXBsZXhpdHk6IDAuNSxcbiAgICAuLi5vcHRpb25zLFxuICB9O1xufVxuXG4vKipcbiAqIENyZWF0ZSBhIHN1Yi1hbnN3ZXJcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN1YkFuc3dlcihcbiAgc3ViUXVlcnlJZDogbnVtYmVyLFxuICBjb250ZW50OiBzdHJpbmcsXG4gIGFnZW50OiBzdHJpbmcsXG4gIG9wdGlvbnM6IFBhcnRpYWw8U3ViQW5zd2VyPiA9IHt9XG4pOiBTdWJBbnN3ZXIge1xuICByZXR1cm4ge1xuICAgIHN1YlF1ZXJ5SWQsXG4gICAgY29udGVudCxcbiAgICBjb25maWRlbmNlOiAwLjgsXG4gICAgYWdlbnQsXG4gICAgbGF0ZW5jeU1zOiAwLFxuICAgIHF1YWxpdHk6IDAuOCxcbiAgICBzdWNjZXNzOiB0cnVlLFxuICAgIC4uLm9wdGlvbnMsXG4gIH07XG59XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBFeHBvcnRzXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG5leHBvcnQgZGVmYXVsdCBSbG1UcmFpbmVyO1xuIl19