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
package edu.snu.reef.dolphin.neuralnet.layers;

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.LayerIndex;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.ActivationFunction;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.NumberOfOutput;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.util.Arrays;

/**
 * Fully connected Layer.
 */
public final class FullyConnectedLayer extends LayerBase {

  private final String activationFunction;

  @Inject
  public FullyConnectedLayer(@Parameter(LayerIndex.class) final int index,
                             @Parameter(ActivationFunction.class) final String activationFunction,
                             @Parameter(NumberOfOutput.class) final int numOutput,
                             final LayerParameterInitializer layerParameterInitializer) {
    super(index, numOutput);
    this.activationFunction = activationFunction;
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return true;
  }

  /** {@inheritDoc} */
  @Override
  public INDArray derivative(final INDArray activation) {
    return Nd4j.getExecutioner().execAndReturn(
        Nd4j.getOpFactory().createTransform(activationFunction, activation.dup()).derivative());
  }

  /**
   * Computes an activation for this fully connected layer.
   * @param input an input value for this layer.
   * @return an activation row vector.
   */
  @Override
  public INDArray feedForward(final INDArray input) {
    // convert input to a row vector.
    final INDArray inputVector = input.reshape(1, input.length());
    // (output row vector) = (input row vector) x (weight matrix) + (bias row vector)
    final INDArray output =
        inputVector.mmul(getLayerParameter().getWeightParam()).addiRowVector(getLayerParameter().getBiasParam());
    // apply activation function.
    return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, output));
  }

  /**
   * Computes an error for this fully connected layer.
   * @param activation an activation row vector.
   * @param nextParameter the parameter of the next layer - the one closer to the output layer.
   * @param nextError the error of the next layer - the one closer to the output layer.
   * @return an error row vector.
   */
  @Override
  public INDArray backPropagate(final INDArray activation,
                                final LayerParameter nextParameter,
                                final INDArray nextError) {
    if (!activation.isRowVector()) {
      throw new RuntimeException(String.format("Invalid activation shape %s. " +
          "An activation for a fully connected layer must be a row vector.", Arrays.toString(activation.shape())));
    }
    // convert a error of the next layer to a row vector.
    final INDArray nextErrorVector = nextError.reshape(1, nextError.length());
    // ((next error row vector) x (weight matrix of the next layer)) * (derivative row vector)
    return nextErrorVector.mmul(nextParameter.getWeightParam().transpose()).muli(derivative(activation));
  }

  /** {@inheritDoc} */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray label) {
    return activation.sub(label);
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final INDArray input, final INDArray error) {
    if (!error.isRowVector()) {
      throw new RuntimeException(String.format("Invalid error shape %s. " +
          "An error of a fully connected layer must be a row vector.", Arrays.toString(error.shape())));
    }
    return LayerParameter.newBuilder()
        .setWeightParam(input.reshape(1, input.length()).transpose().mmul(error))
        .setBiasParam(error)
        .build();
  }
}
