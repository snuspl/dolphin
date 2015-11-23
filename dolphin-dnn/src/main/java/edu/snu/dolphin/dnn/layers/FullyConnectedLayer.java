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

import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.LayerIndex;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.NumberOfOutput;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.util.Arrays;

/**
 * Fully connected Layer.
 */
public final class FullyConnectedLayer extends LayerBase {

  @Inject
  public FullyConnectedLayer(@Parameter(LayerIndex.class) final int index,
                             @Parameter(NumberOfOutput.class) final int numOutput,
                             final LayerParameterInitializer layerParameterInitializer) {
    super(index, numOutput);
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return true;
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
    return inputVector.mmul(getLayerParameter().getWeightParam()).addiRowVector(getLayerParameter().getBiasParam());
  }

  /**
   * Computes an error for this fully connected layer.
   * @param input the input value.
   * @param activation the activation value.
   * @param nextError the error of the next layer - the one closer to the output layer.
   * @return an error for this layer.
   */
  @Override
  public INDArray backPropagate(final INDArray input, final INDArray activation, final INDArray nextError) {
    // convert a error of the next layer to a row vector.
    final INDArray nextErrorVector = nextError.reshape(1, nextError.length());
    // (next error row vector) x (weight matrix of this layer)
    return nextErrorVector.mmul(getLayerParameter().getWeightParam().transpose());
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
