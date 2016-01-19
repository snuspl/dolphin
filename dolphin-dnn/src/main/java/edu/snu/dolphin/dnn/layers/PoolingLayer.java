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
import edu.snu.dolphin.dnn.blas.function.Function;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

/**
 * Pooling layer.
 *
 * This layer is not learnable.
 * This layer resizes input matrix spatially, using max pooling or average pooling.
 * In a forward pass,
 * max pooling picks maximum value in certain range ( kernalHeight * kernalWidth) and these values make up output.
 * average pooling get average of values in certain range (kernalHeight * kernalWidth) and these values make up output.
 * In a backward pass,
 * each value of error matrix is the sum of elements in next error matrix
 * on which had an impact in feedforward step.
 */

public abstract class PoolingLayer extends LayerBase {

  private final int[] outputShape;
  private Matrix trackMatrix;
  private Function poolingFunctions;

  @Inject
  public PoolingLayer(@Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
                      @Parameter(LayerConfigurationParameters.LayerInputShape.class) final String inputShape,
                      final LayerParameterInitializer layerParameterInitializer) {
    super(index, inputShape);
    this.outputShape = setOutputShape();
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  @Override
  public int[] getOutputShape() {
    return outputShape;
  }

  /**
   * This function computes output size.
   * input size: row * col
   * output size: row' * col'
   * row = (row − kernal_height) / stride + 1
   * col = (col − kernal_width) / stride + 1
   */

  public int[] setOutputShape() {
    final int[] inputShape = getInputShape();
    int[] computedShape;
    switch (inputShape.length) {
    case 1:
      computedShape = new int[1];
      computedShape[0] = (inputShape[0] - getLayerParameter().getKernalHeight()) / getLayerParameter().getStride() + 1;
      return computedShape;
    case 2:
      computedShape = new int[2];
      computedShape[0] = (inputShape[0] - getLayerParameter().getKernalHeight()) / getLayerParameter().getStride() + 1;
      computedShape[1] = (inputShape[1] - getLayerParameter().getKernalWidth()) / getLayerParameter().getStride() + 1;
      return computedShape;
    default:
      throw new IllegalArgumentException("Unsupported input dimension: " + Integer.toString(inputShape.length));
    }
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return false;
  }

  public abstract Matrix feedForwardMaxPooling(final Matrix input);

  public abstract Matrix feedForwardAveragePooling(final Matrix input);

  /**
   * Computes output values for this pooling layer.
   * available pooling type: max, average
   * @param input input values for this layer.
   * @return output values for this layer.
   */

  @Override
  public  Matrix feedForward(final Matrix input) {
    switch (getLayerParameter().getPoolingType().toLowerCase()) {
    case "max" :
      return feedForwardMaxPooling(input);
    case "average" :
      return feedForwardAveragePooling(input);
    default:
      throw new IllegalArgumentException("Illegal pooling type: " + getLayerParameter().getPoolingType());
    }
  }

  public abstract Matrix backPropagateMaxPooling(final Matrix input, final Matrix nextError);

  public abstract Matrix backPropagateAveragePooling(final Matrix input, final Matrix nextError);

  /**
   * Computes errors for this pooling layer.
   * available pooling type: max, average
   * @param input the input values for this layer.
   * @param activation the output values.
   * @param nextError the errors of the next layer - the one closer to the output layer.
   * @return errors for this layer with the specified input value.
   */

  @Override
  public Matrix backPropagate(final Matrix input, final Matrix activation, final Matrix nextError) {
    switch (getLayerParameter().getPoolingType().toLowerCase()) {
    case "max" :
      return backPropagateMaxPooling(input, nextError);
    case "average" :
      return backPropagateAveragePooling(input, nextError);
    default:
      throw new IllegalArgumentException("Illegal pooling type: " + getLayerParameter().getPoolingType());
    }
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final Matrix input, final Matrix error) {
    throw new RuntimeException("This layer doesn't have parameter gradient");
  }
}
