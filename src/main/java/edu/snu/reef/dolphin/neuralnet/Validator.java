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

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.util.List;

/**
 * Class for validating a neural network model using a given data input.
 * Calculates the prediction accuracy for the given validation data set.
 */
final class Validator {
  private final NeuralNetwork network;
  private int totalNum;
  private int correctNum;

  @Inject
  Validator(final NeuralNetwork network) {
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
