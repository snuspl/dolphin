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
package edu.snu.dolphin.dnn.layers;

import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.ActivationFunction;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.LayerIndex;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.NumberOfOutput;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Activation Layer.
 */
public final class ActivationLayer extends LayerBase {

  private final String activationFunction;

  @Inject
  public ActivationLayer(@Parameter(LayerIndex.class) final int index,
                         @Parameter(ActivationFunction.class) final String activationFunction,
                         @Parameter(NumberOfOutput.class) final int numOutput) {
    super(index, numOutput);
    this.activationFunction = activationFunction;
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return false;
  }

  /**
   * Applies the specified activation function.
   * @param input an input value for this layer.
   * @return the activation.
   */
  @Override
  public INDArray feedForward(final INDArray input) {
    // apply activation function.
    return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, input.dup()));
  }

  /**
   * Computes an error for this activation layer.
   * @param input the input value.
   * @param activation the activation value.
   * @param nextError an error of the next layer - the one closer to the output layer.
   * @return an error for this activation layer.
   */
  @Override
  public INDArray backPropagate(final INDArray input, final INDArray activation, final INDArray nextError) {
    final INDArray derivative = Nd4j.getExecutioner().execAndReturn(
        Nd4j.getOpFactory().createTransform(activationFunction, activation.dup()).derivative());
    return nextError.mul(derivative);
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final INDArray input, final INDArray error) {
    throw new RuntimeException("This layer is not learnable");
  }
}
