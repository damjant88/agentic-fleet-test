"use strict";
/**
 * RLM - Retrieval Language Model
 *
 * A recursive retrieval-augmented generation system that combines
 * memory search with intelligent query decomposition and synthesis.
 *
 * @example Basic Usage
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   maxDepth: 3,
 *   enableCache: true,
 *   retrievalTopK: 10,
 * });
 *
 * // Add knowledge
 * await rlm.addMemory('TypeScript adds static typing to JavaScript.');
 * await rlm.addMemory('React is a library for building user interfaces.');
 *
 * // Query with retrieval
 * const answer = await rlm.query('Compare TypeScript and JavaScript');
 * console.log(answer.text);
 * console.log('Confidence:', answer.confidence);
 * console.log('Sources:', answer.sources.length);
 * ```
 *
 * @example Streaming
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController();
 *
 * for await (const event of rlm.queryStream('Explain machine learning')) {
 *   if (event.type === 'token') {
 *     process.stdout.write(event.text);
 *   } else {
 *     console.log('\n\nQuality:', event.answer.qualityScore);
 *   }
 * }
 * ```
 *
 * @example With Reflection
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   enableReflection: true,
 *   maxReflectionIterations: 2,
 *   minQualityScore: 0.8,
 * });
 *
 * // Answers will be iteratively refined until quality >= 0.8
 * const answer = await rlm.query('Complex technical question...');
 * ```
 *
 * @module rlm
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createSubAnswer = exports.createSubQuery = exports.createEmptyExample = exports.createRlmTrainer = exports.RlmTrainer = exports.HARD_NEGATIVE_PAIRS = exports.AGENT_DEFINITIONS = exports.ROUTING_FOCUSED_CONFIG = exports.THOROUGH_RLM_CONFIG = exports.FAST_RLM_CONFIG = exports.DEFAULT_RLM_CONFIG = exports.RlmController = void 0;
// Export all types
__exportStar(require("./types"), exports);
// Export the controller
var controller_1 = require("./controller");
Object.defineProperty(exports, "RlmController", { enumerable: true, get: function () { return controller_1.RlmController; } });
// Export training module
var training_1 = require("./training");
// Constants
Object.defineProperty(exports, "DEFAULT_RLM_CONFIG", { enumerable: true, get: function () { return training_1.DEFAULT_RLM_CONFIG; } });
Object.defineProperty(exports, "FAST_RLM_CONFIG", { enumerable: true, get: function () { return training_1.FAST_RLM_CONFIG; } });
Object.defineProperty(exports, "THOROUGH_RLM_CONFIG", { enumerable: true, get: function () { return training_1.THOROUGH_RLM_CONFIG; } });
Object.defineProperty(exports, "ROUTING_FOCUSED_CONFIG", { enumerable: true, get: function () { return training_1.ROUTING_FOCUSED_CONFIG; } });
Object.defineProperty(exports, "AGENT_DEFINITIONS", { enumerable: true, get: function () { return training_1.AGENT_DEFINITIONS; } });
Object.defineProperty(exports, "HARD_NEGATIVE_PAIRS", { enumerable: true, get: function () { return training_1.HARD_NEGATIVE_PAIRS; } });
// Classes
Object.defineProperty(exports, "RlmTrainer", { enumerable: true, get: function () { return training_1.RlmTrainer; } });
// Factory functions
Object.defineProperty(exports, "createRlmTrainer", { enumerable: true, get: function () { return training_1.createRlmTrainer; } });
Object.defineProperty(exports, "createEmptyExample", { enumerable: true, get: function () { return training_1.createEmptyExample; } });
Object.defineProperty(exports, "createSubQuery", { enumerable: true, get: function () { return training_1.createSubQuery; } });
Object.defineProperty(exports, "createSubAnswer", { enumerable: true, get: function () { return training_1.createSubAnswer; } });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi9zcmMvcmxtL2luZGV4LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFBQTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBeURHOzs7Ozs7Ozs7Ozs7Ozs7OztBQUVILG1CQUFtQjtBQUNuQiwwQ0FBd0I7QUFFeEIsd0JBQXdCO0FBQ3hCLDJDQUE2QztBQUFwQywyR0FBQSxhQUFhLE9BQUE7QUFFdEIseUJBQXlCO0FBQ3pCLHVDQTZCb0I7QUFoQmxCLFlBQVk7QUFDWiw4R0FBQSxrQkFBa0IsT0FBQTtBQUNsQiwyR0FBQSxlQUFlLE9BQUE7QUFDZiwrR0FBQSxtQkFBbUIsT0FBQTtBQUNuQixrSEFBQSxzQkFBc0IsT0FBQTtBQUN0Qiw2R0FBQSxpQkFBaUIsT0FBQTtBQUNqQiwrR0FBQSxtQkFBbUIsT0FBQTtBQUVuQixVQUFVO0FBQ1Ysc0dBQUEsVUFBVSxPQUFBO0FBRVYsb0JBQW9CO0FBQ3BCLDRHQUFBLGdCQUFnQixPQUFBO0FBQ2hCLDhHQUFBLGtCQUFrQixPQUFBO0FBQ2xCLDBHQUFBLGNBQWMsT0FBQTtBQUNkLDJHQUFBLGVBQWUsT0FBQSIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogUkxNIC0gUmV0cmlldmFsIExhbmd1YWdlIE1vZGVsXG4gKlxuICogQSByZWN1cnNpdmUgcmV0cmlldmFsLWF1Z21lbnRlZCBnZW5lcmF0aW9uIHN5c3RlbSB0aGF0IGNvbWJpbmVzXG4gKiBtZW1vcnkgc2VhcmNoIHdpdGggaW50ZWxsaWdlbnQgcXVlcnkgZGVjb21wb3NpdGlvbiBhbmQgc3ludGhlc2lzLlxuICpcbiAqIEBleGFtcGxlIEJhc2ljIFVzYWdlXG4gKiBgYGB0eXBlc2NyaXB0XG4gKiBpbXBvcnQgeyBSbG1Db250cm9sbGVyIH0gZnJvbSAnQHJ1dmVjdG9yL3J1dmxsbSc7XG4gKlxuICogY29uc3QgcmxtID0gbmV3IFJsbUNvbnRyb2xsZXIoe1xuICogICBtYXhEZXB0aDogMyxcbiAqICAgZW5hYmxlQ2FjaGU6IHRydWUsXG4gKiAgIHJldHJpZXZhbFRvcEs6IDEwLFxuICogfSk7XG4gKlxuICogLy8gQWRkIGtub3dsZWRnZVxuICogYXdhaXQgcmxtLmFkZE1lbW9yeSgnVHlwZVNjcmlwdCBhZGRzIHN0YXRpYyB0eXBpbmcgdG8gSmF2YVNjcmlwdC4nKTtcbiAqIGF3YWl0IHJsbS5hZGRNZW1vcnkoJ1JlYWN0IGlzIGEgbGlicmFyeSBmb3IgYnVpbGRpbmcgdXNlciBpbnRlcmZhY2VzLicpO1xuICpcbiAqIC8vIFF1ZXJ5IHdpdGggcmV0cmlldmFsXG4gKiBjb25zdCBhbnN3ZXIgPSBhd2FpdCBybG0ucXVlcnkoJ0NvbXBhcmUgVHlwZVNjcmlwdCBhbmQgSmF2YVNjcmlwdCcpO1xuICogY29uc29sZS5sb2coYW5zd2VyLnRleHQpO1xuICogY29uc29sZS5sb2coJ0NvbmZpZGVuY2U6JywgYW5zd2VyLmNvbmZpZGVuY2UpO1xuICogY29uc29sZS5sb2coJ1NvdXJjZXM6JywgYW5zd2VyLnNvdXJjZXMubGVuZ3RoKTtcbiAqIGBgYFxuICpcbiAqIEBleGFtcGxlIFN0cmVhbWluZ1xuICogYGBgdHlwZXNjcmlwdFxuICogaW1wb3J0IHsgUmxtQ29udHJvbGxlciB9IGZyb20gJ0BydXZlY3Rvci9ydXZsbG0nO1xuICpcbiAqIGNvbnN0IHJsbSA9IG5ldyBSbG1Db250cm9sbGVyKCk7XG4gKlxuICogZm9yIGF3YWl0IChjb25zdCBldmVudCBvZiBybG0ucXVlcnlTdHJlYW0oJ0V4cGxhaW4gbWFjaGluZSBsZWFybmluZycpKSB7XG4gKiAgIGlmIChldmVudC50eXBlID09PSAndG9rZW4nKSB7XG4gKiAgICAgcHJvY2Vzcy5zdGRvdXQud3JpdGUoZXZlbnQudGV4dCk7XG4gKiAgIH0gZWxzZSB7XG4gKiAgICAgY29uc29sZS5sb2coJ1xcblxcblF1YWxpdHk6JywgZXZlbnQuYW5zd2VyLnF1YWxpdHlTY29yZSk7XG4gKiAgIH1cbiAqIH1cbiAqIGBgYFxuICpcbiAqIEBleGFtcGxlIFdpdGggUmVmbGVjdGlvblxuICogYGBgdHlwZXNjcmlwdFxuICogaW1wb3J0IHsgUmxtQ29udHJvbGxlciB9IGZyb20gJ0BydXZlY3Rvci9ydXZsbG0nO1xuICpcbiAqIGNvbnN0IHJsbSA9IG5ldyBSbG1Db250cm9sbGVyKHtcbiAqICAgZW5hYmxlUmVmbGVjdGlvbjogdHJ1ZSxcbiAqICAgbWF4UmVmbGVjdGlvbkl0ZXJhdGlvbnM6IDIsXG4gKiAgIG1pblF1YWxpdHlTY29yZTogMC44LFxuICogfSk7XG4gKlxuICogLy8gQW5zd2VycyB3aWxsIGJlIGl0ZXJhdGl2ZWx5IHJlZmluZWQgdW50aWwgcXVhbGl0eSA+PSAwLjhcbiAqIGNvbnN0IGFuc3dlciA9IGF3YWl0IHJsbS5xdWVyeSgnQ29tcGxleCB0ZWNobmljYWwgcXVlc3Rpb24uLi4nKTtcbiAqIGBgYFxuICpcbiAqIEBtb2R1bGUgcmxtXG4gKi9cblxuLy8gRXhwb3J0IGFsbCB0eXBlc1xuZXhwb3J0ICogZnJvbSAnLi90eXBlcyc7XG5cbi8vIEV4cG9ydCB0aGUgY29udHJvbGxlclxuZXhwb3J0IHsgUmxtQ29udHJvbGxlciB9IGZyb20gJy4vY29udHJvbGxlcic7XG5cbi8vIEV4cG9ydCB0cmFpbmluZyBtb2R1bGVcbmV4cG9ydCB7XG4gIC8vIFR5cGVzXG4gIHR5cGUgRGVjb21wb3NpdGlvblN0cmF0ZWd5LFxuICB0eXBlIFN1YlF1ZXJ5LFxuICB0eXBlIFF1ZXJ5RGVjb21wb3NpdGlvbixcbiAgdHlwZSBTdWJBbnN3ZXIsXG4gIHR5cGUgUmxtVHJhamVjdG9yeU1ldGFkYXRhLFxuICB0eXBlIFJsbVRyYWluaW5nRXhhbXBsZSxcbiAgdHlwZSBDb250cmFzdGl2ZVBhaXIsXG4gIHR5cGUgUmxtVHJhaW5pbmdDb25maWcsXG4gIHR5cGUgVHJhaW5pbmdSZXN1bHQgYXMgUmxtVHJhaW5pbmdSZXN1bHQsXG4gIHR5cGUgRXZhbHVhdGlvblJlc3VsdCBhcyBSbG1FdmFsdWF0aW9uUmVzdWx0LFxuXG4gIC8vIENvbnN0YW50c1xuICBERUZBVUxUX1JMTV9DT05GSUcsXG4gIEZBU1RfUkxNX0NPTkZJRyxcbiAgVEhPUk9VR0hfUkxNX0NPTkZJRyxcbiAgUk9VVElOR19GT0NVU0VEX0NPTkZJRyxcbiAgQUdFTlRfREVGSU5JVElPTlMsXG4gIEhBUkRfTkVHQVRJVkVfUEFJUlMsXG5cbiAgLy8gQ2xhc3Nlc1xuICBSbG1UcmFpbmVyLFxuXG4gIC8vIEZhY3RvcnkgZnVuY3Rpb25zXG4gIGNyZWF0ZVJsbVRyYWluZXIsXG4gIGNyZWF0ZUVtcHR5RXhhbXBsZSxcbiAgY3JlYXRlU3ViUXVlcnksXG4gIGNyZWF0ZVN1YkFuc3dlcixcbn0gZnJvbSAnLi90cmFpbmluZyc7XG4iXX0=