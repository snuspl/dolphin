/*
 * Copyright (C) 2015 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.snu.reef.dolphin.neuralnet;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The task that trains neural network with the data set.
 *
 * Assumes the input file can be parsed by NeuralNetworkDataParser.
 */
@TaskSide
public final class NeuralNetworkTask implements Task {

  private static final Logger LOG = Logger.getLogger(NeuralNetworkTask.class.getName());

  private final Validator crossValidator;
  private final Validator trainingValidator;
  private final DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> dataParser;
  private final NeuralNetwork neuralNetwork;
  private final int maxIterations;

  /**
   * Class for validation of neural network model.
   * Calculates the prediction accuracy for validation data set.
   */
  private static final class Validator {
    private final NeuralNetwork network;
    private int totalNum;
    private int correctNum;

    @Inject
    private Validator(final NeuralNetwork network) {
      this.network = network;
    }

    public void validate(final INDArray input, final int label) {
      final List<INDArray> activations = network.feedForward(input);
      final INDArray output = activations.get(activations.size() - 1);
      float maxValue = output.getFloat(0);

      // Find the index with highest probability.
      int maxIndex = 0;
      for (int i = 1; i < output.length(); ++i) {
        if (output.getFloat(i) > maxValue) {
          maxValue = output.getFloat(i);
          maxIndex = i;
        }
      }

      ++totalNum;
      if (maxIndex == label) {
        ++correctNum;
      }
    }

    /**
     * Reset statistics.
     */
    public void reset() {
      totalNum = 0;
      correctNum = 0;
    }

    /**
     * @return the prediction accuracy of model.
     */
    public float getAccuracy() {
      return correctNum / (float) totalNum;
    }

    /**
     * @return the prediction error of model.
     */
    public float getError() {
      return 1 - getAccuracy();
    }

    /**
     * @return the total number of samples that are used for evaluation.
     */
    public int getTotalNum() {
      return totalNum;
    }
  }

  @Inject
  NeuralNetworkTask(final DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> dataParser,
                    final NeuralNetwork neuralNetwork,
                    @Parameter(MaxIterations.class) final int maxIterations) {
    super();
    this.dataParser = dataParser;
    this.neuralNetwork = neuralNetwork;
    this.maxIterations = maxIterations;
    this.trainingValidator = new Validator(neuralNetwork);
    this.crossValidator = new Validator(neuralNetwork);
  }

  /** {@inheritDoc} */
  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "ComputeTask.call() commencing....");

    final List<Pair<Pair<INDArray, Integer>, Boolean>> dataSet = dataParser.get();

    for (int i = 0; i < maxIterations; ++i) {
      for (final Pair<Pair<INDArray, Integer>, Boolean> data : dataSet) {
        final INDArray input = data.getFirst().getFirst();
        final int label = data.getFirst().getSecond();
        final boolean isValidation = data.getSecond();
        if (isValidation) {
          crossValidator.validate(input, label);
        } else {
          neuralNetwork.train(input, label);
          trainingValidator.validate(input, label);
        }
      }
      LOG.log(Level.INFO, "=========================================================");
      LOG.log(Level.INFO, "Iteration: {0}", String.valueOf(i));
      LOG.log(Level.INFO, "Training Error: {0}", String.valueOf(trainingValidator.getError()));
      LOG.log(Level.INFO, "Cross Validation Error: {0}", String.valueOf(crossValidator.getError()));
      LOG.log(Level.INFO, "# of validation inputs: {0}", String.valueOf(crossValidator.getTotalNum()));
      LOG.log(Level.INFO, "=========================================================");
      crossValidator.reset();
      trainingValidator.reset();
    }

    return null;
  }
}
