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
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.io.network.util.Pair;
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

  private final Evaluator evaluator;
  private final DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> dataParser;
  private final NeuralNetwork neuralNetwork;

  /**
   * Class for calculating the prediction accuracy for validation data set.
   */
  private static class Evaluator {
    private int totalNum;
    private int correctNum;

    public void evaluate(final INDArray output, final int expected) {
      ++totalNum;

      double maxValue = output.getDouble(0);
      int maxIndex = 0;
      for (int i = 1; i < output.length(); ++i) {
        if (output.getDouble(i) > maxValue) {
          maxValue = output.getDouble(i);
          maxIndex = i;
        }
      }
      if (maxIndex == expected) {
        ++correctNum;
      }
    }

    /**
     * @return the prediction accuracy of model.
     */
    public double getStats() {
      return correctNum / (double) totalNum;
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
                    final NeuralNetwork neuralNetwork) {
    super();
    this.dataParser = dataParser;
    this.neuralNetwork = neuralNetwork;
    this.evaluator = new Evaluator();
  }

  /** {@inheritDoc} */
  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "ComputeTask.call() commencing....");

    final List<Pair<Pair<INDArray, Integer>, Boolean>> dataSet = dataParser.get();
    for (final Pair<Pair<INDArray, Integer>, Boolean> data : dataSet) {
      final INDArray input = data.getFirst().getFirst();
      final int label = data.getFirst().getSecond();
      final boolean isValidation = data.getSecond();
      if (isValidation) {
        final List<INDArray> activations = neuralNetwork.feedForward(input);
        evaluator.evaluate(activations.get(activations.size() - 1), label);
      } else {
        neuralNetwork.train(input, label);
      }
    }

    LOG.log(Level.INFO, "=========================================================");
    LOG.log(Level.INFO, "Result: {0}", String.valueOf(evaluator.getStats()));
    LOG.log(Level.INFO, "# of validation inputs: {0}", String.valueOf(evaluator.getTotalNum()));
    LOG.log(Level.INFO, "=========================================================");

    return null;
  }
}
