import * as tf from '@tensorflow/tfjs';

// Type definitions
export type TypeBoard = number[][];

export interface ModelPrediction {
  move: number;          // 0=up, 1=right, 2=down, 3=left
  confidence: number;    // Probability of the selected move (0-1)
  value: number;         // Estimated value of the board state
  policy: number[];      // Full policy distribution [up, right, down, left]
}

export interface ModelLoadResult {
  success: boolean;
  error?: string;
  modelInfo?: {
    inputs: tf.SymbolicTensor[];
    outputs: tf.SymbolicTensor[];
    parameters: number;
  };
}

export enum ModelState {
  UNLOADED = 'unloaded',
  LOADING = 'loading', 
  LOADED = 'loaded',
  ERROR = 'error'
}

/**
 * CNN2048Model class for AI-powered 2048 gameplay
 * 
 * This class handles loading a trained TensorFlow.js model and performing
 * inference to predict optimal moves for 2048 game boards.
 */
export class CNN2048Model {
  private model: tf.LayersModel | null = null;
  private state: ModelState = ModelState.UNLOADED;
  private errorMessage: string | null = null;

  /**
   * Get the current state of the model
   */
  public getState(): ModelState {
    return this.state;
  }

  /**
   * Get the last error message if any
   */
  public getError(): string | null {
    return this.errorMessage;
  }

  /**
   * Check if the model is ready for predictions
   */
  public isReady(): boolean {
    return this.state === ModelState.LOADED && this.model !== null;
  }

  /**
   * Load the TensorFlow.js model from the public directory
   * 
   * @param modelPath - Path to the model.json file (default: '/model/model.json')
   * @returns Promise resolving to load result with success/error information
   */
  public async loadModel(modelPath: string = '/model/model.json'): Promise<ModelLoadResult> {
    this.state = ModelState.LOADING;
    this.errorMessage = null;

    try {
      console.log(`🤖 Loading AI model from: ${modelPath}`);
      
      // Load the model
      this.model = await tf.loadLayersModel(modelPath);
      
      // Validate model architecture
      const modelInfo = this.validateModelArchitecture();
      if (!modelInfo.success) {
        throw new Error(modelInfo.error || 'Model validation failed');
      }

      this.state = ModelState.LOADED;
      console.log('✅ AI model loaded successfully');
      
      return {
        success: true,
        modelInfo: {
          inputs: this.model.inputs,
          outputs: this.model.outputs,
          parameters: this.model.countParams()
        }
      };

    } catch (error) {
      this.state = ModelState.ERROR;
      this.errorMessage = error instanceof Error ? error.message : 'Unknown error loading model';
      
      console.error('❌ Failed to load AI model:', this.errorMessage);
      
      return {
        success: false,
        error: this.errorMessage
      };
    }
  }

  /**
   * Validate that the loaded model has the expected architecture
   * 
   * @returns Validation result with success/error information
   */
  private validateModelArchitecture(): { success: boolean; error?: string } {
    if (!this.model) {
      return { success: false, error: 'No model loaded' };
    }

    try {
      // Check input shape: should be [null, 4, 4, 16] for one-hot encoded board
      const inputShape = this.model.inputs[0].shape;
      if (inputShape.length !== 4 || inputShape[1] !== 4 || inputShape[2] !== 4 || inputShape[3] !== 16) {
        return {
          success: false,
          error: `Invalid input shape: expected [null, 4, 4, 16], got [${inputShape.join(', ')}]`
        };
      }

      // Check outputs: should have policy (4 moves) and value (1 scalar)
      if (this.model.outputs.length < 1) {
        return {
          success: false,
          error: 'Model should have at least 1 output (policy or combined output)'
        };
      }

      console.log(`✅ Model validation passed - Input: [${inputShape.join(', ')}], Outputs: ${this.model.outputs.length}`);
      return { success: true };

    } catch (error) {
      return {
        success: false,
        error: `Model validation error: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Encode a 2D board array into a one-hot encoded tensor
   * 
   * @param board - 4x4 array of numbers representing the game board
   * @returns Encoded tensor with shape [1, 4, 4, 16] or null if encoding fails
   */
  public encodeBoard(board: TypeBoard): tf.Tensor4D | null {
    try {
      // Validate input board dimensions
      if (!board || board.length !== 4 || !board.every(row => row.length === 4)) {
        console.error('❌ Invalid board dimensions: expected 4x4 array');
        return null;
      }

      // Convert board values to indices for one-hot encoding
      // 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, 16 -> 4, ..., 2048 -> 11, etc.
      const indices: number[][] = board.map(row => 
        row.map(cell => {
          if (cell === 0) return 0;
          const logValue = Math.log2(cell);
          // Ensure we don't exceed our 16 channels (0-15)
          return Math.min(Math.max(Math.round(logValue), 0), 15);
        })
      );

      // Create tensor from indices with shape [4, 4]
      const indicesTensor = tf.tensor2d(indices, [4, 4]);
      
      // Convert to one-hot encoding with 16 channels
      // This creates shape [4, 4, 16]
      const oneHotTensor = tf.oneHot(indicesTensor, 16);
      
      // Add batch dimension to get shape [1, 4, 4, 16]
      const batchedTensor = tf.expandDims(oneHotTensor, 0) as tf.Tensor4D;
      
      // Clean up intermediate tensors to prevent memory leaks
      indicesTensor.dispose();
      oneHotTensor.dispose();
      
      return batchedTensor;

    } catch (error) {
      console.error('❌ Error encoding board:', error);
      return null;
    }
  }

  /**
   * Predict the best move for a given board state
   * 
   * @param board - 4x4 array representing the current game board
   * @param validMoves - Array of valid move indices (0=up, 1=right, 2=down, 3=left)
   * @returns Promise resolving to prediction with move, confidence, and value
   */
  public async predictMove(board: TypeBoard, validMoves: number[]): Promise<ModelPrediction> {
    // Check if model is ready
    if (!this.isReady() || !this.model) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // Validate inputs
    if (!validMoves || validMoves.length === 0) {
      throw new Error('No valid moves provided');
    }

    if (validMoves.some(move => move < 0 || move > 3)) {
      throw new Error('Invalid move indices. Moves must be 0-3 (up, right, down, left)');
    }

    try {
      // Encode the board
      const encodedBoard = this.encodeBoard(board);
      if (!encodedBoard) {
        throw new Error('Failed to encode board');
      }

      // Run model prediction
      const predictions = this.model.predict(encodedBoard) as tf.Tensor | tf.Tensor[];
      
      let policyTensor: tf.Tensor;
      let valueTensor: tf.Tensor;

      // Handle different model output formats
      if (Array.isArray(predictions)) {
        // Multi-output model: [policy, value]
        if (predictions.length < 2) {
          throw new Error('Expected at least 2 outputs from model (policy and value)');
        }
        policyTensor = predictions[0];
        valueTensor = predictions[1];
      } else {
        // Single output model: assume it contains both policy and value
        // Split the output tensor (first 4 values = policy, last value = value)
        const flatOutput = tf.reshape(predictions, [-1]);
        policyTensor = tf.slice(flatOutput, [0], [4]);
        valueTensor = tf.slice(flatOutput, [4], [1]);
        flatOutput.dispose();
      }

      // Extract policy and value data
      const policyData = await policyTensor.data();
      const valueData = await valueTensor.data();
      
      // Convert to regular arrays
      const policy = Array.from(policyData);
      const value = valueData[0];

      // Apply move masking - set invalid moves to very low probability
      const maskedPolicy = policy.map((prob, index) => 
        validMoves.includes(index) ? prob : -Infinity
      );

      // Apply softmax to get normalized probabilities for valid moves only
      const maxLogit = Math.max(...maskedPolicy.filter(p => p !== -Infinity));
      const expValues = maskedPolicy.map(logit => 
        logit === -Infinity ? 0 : Math.exp(logit - maxLogit)
      );
      const sumExp = expValues.reduce((sum, val) => sum + val, 0);
      const softmaxPolicy = expValues.map(val => val / sumExp);

      // Find the best valid move
      let bestMove = validMoves[0];
      let bestConfidence = softmaxPolicy[bestMove];

      for (const move of validMoves) {
        if (softmaxPolicy[move] > bestConfidence) {
          bestMove = move;
          bestConfidence = softmaxPolicy[move];
        }
      }

      // Clean up tensors to prevent memory leaks
      encodedBoard.dispose();
      policyTensor.dispose();
      valueTensor.dispose();
      
      if (Array.isArray(predictions)) {
        predictions.forEach(tensor => tensor.dispose());
      } else {
        predictions.dispose();
      }

      return {
        move: bestMove,
        confidence: bestConfidence,
        value: value,
        policy: softmaxPolicy
      };

    } catch (error) {
      console.error('❌ Error during prediction:', error);
      throw new Error(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Dispose of the loaded model to free memory
   */
  public dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.state = ModelState.UNLOADED;
    this.errorMessage = null;
    console.log('🗑️ AI model disposed');
  }

  /**
   * Get model information for debugging
   */
  public getModelInfo(): object | null {
    if (!this.model) return null;

    return {
      state: this.state,
      inputShape: this.model.inputs[0].shape,
      outputShapes: this.model.outputs.map(output => output.shape),
      parameters: this.model.countParams(),
      backend: tf.getBackend()
    };
  }
}

// Export a singleton instance for use across the application
export const aiModel = new CNN2048Model(); 