/*
 * Copyright (C) 2016 Seoul National University
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
   * Convolutional layer.
   *
   * This layer is learnable having the updatable parameter (weight and bias).
   * This layer works for only 2D inputs.
   * In a forward pass,
   * computes the product between weight and the input within kernel range and produce activation matrix.
   * In a backward pass,
   * error of each input pixel comes from
   * the product between weight and errors of output pixels affected by the input pixel in feedforward step.
   */
public final class ConvolutionalLayer extends LayerBase {

  private final int[] outputShape;
  private final int paddingHeight;
  private final int paddingWidth;
  private final int strideHeight;
  private final int strideWidth;
  private final int kernelHeight;
  private final int kernelWidth;

  @Inject
  private ConvolutionalLayer(@Parameter(LayerIndex.class) final int index,
                             @Parameter(LayerInputShape.class) final String inputShape,
                             @Parameter(StrideHeight.class) final int paddingHeight,
                             @Parameter(StrideWidth.class) final int paddingWidth,
                             @Parameter(StrideHeight.class) final int strideHeight,
                             @Parameter(StrideWidth.class) final int strideWidth,
                             @Parameter(KernelHeight.class) final int kernelHeight,
                             @Parameter(KernelWidth.class) final int kernelWidth,
                             final LayerParameterInitializer layerParameterInitializer) {
    super(index, inputShape);
    this.paddingHeight = paddingHeight;
    this.paddingWidth = paddingWidth;
    this.strideHeight = strideHeight;
    this.strideWidth = strideWidth;
    this.kernelHeight = kernelHeight;
    this.kernelWidth = kernelWidth;
    this.outputShape = layerParameterInitializer.getOutputShape();
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  @Override
  public int[] getOutputShape() {
    return outputShape;
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return false;
  }

  /**
   * Computes output values for this convolutional layer.
   * @param input input values for this layer.
   * @return output values for this layer.
   */
  @Override
  public Matrix feedForward(final Matrix input) {
    throw new RuntimeException("Not Implemented");
  }

  /**
   * Computes errors for this convolutional layer.
   * @param input the input values for this layer.
   * @param activation the output values.
   * @param nextError the errors of the next layer - the one closer to the output layer.
   * @return errors for this layer with the specified input value.
   */
  @Override
  public Matrix backPropagate(final Matrix input, final Matrix activation, final Matrix nextError) {
    throw new RuntimeException("Not Implemented");
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
