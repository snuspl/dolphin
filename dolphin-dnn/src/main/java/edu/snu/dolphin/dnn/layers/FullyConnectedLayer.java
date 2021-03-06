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

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.*;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

/**
 * Fully connected layer.
 *
 * This layer is learnable having the updatable parameter (weight and bias).
 * This layer computes the inner product between an input and its weight matrix,
 * and adds the bias vector to its output.
 */
public final class FullyConnectedLayer extends LayerBase {

  private final int[] outputShape;

  @Inject
  private FullyConnectedLayer(@Parameter(LayerIndex.class) final int index,
                              @Parameter(LayerInputShape.class) final String inputShape,
                              final LayerParameterInitializer layerParameterInitializer) {
    super(index, inputShape);
    this.outputShape = layerParameterInitializer.getOutputShape();
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  /** {@inheritDoc} */
  @Override
  public int[] getOutputShape() {
    return outputShape;
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return true;
  }

  /**
   * Computes output values for this fully connected layer.
   * @param input input values for this layer.
   * @return output values for this layer.
   */
  @Override
  public Matrix feedForward(final Matrix input) {
    // (output matrix) = (weight matrix) x (input matrix) + (bias column vector)
    return getLayerParameter().getWeightParam().mmul(input).addiColumnVector(getLayerParameter().getBiasParam());
  }

  /**
   * Computes errors for this fully connected layer.
   * @param input the input values for this layer.
   * @param activation the output values.
   * @param nextError the errors of the next layer - the one closer to the output layer.
   * @return errors for this layer with the specified input value.
   */
  @Override
  public Matrix backPropagate(final Matrix input, final Matrix activation, final Matrix nextError) {
    // (error matrix) = (transposed weight matrix) x (next error matrix)
    return getLayerParameter().getWeightParam().transpose().mmul(nextError);
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final Matrix input, final Matrix error) {
    return LayerParameter.newBuilder()
        .setWeightParam(error.mmul(input.transpose()))
        .setBiasParam(error.rowSums())
        .build();
  }
}
