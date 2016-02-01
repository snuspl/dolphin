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
package edu.snu.dolphin.dnn.layerparam.initializer;

import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

import static edu.snu.dolphin.dnn.util.NeuralNetworkUtils.shapeFromString;

/**
 * Pooling Layer parameter initializer.
 *
 * This initializer is for pooling layers which do not have layer parameters.
 */
public final class PoolingLayerParameterInitializer implements LayerParameterInitializer {

  private final int index;
  private final int[] inputShape;
  private final int[] outputShape;
  private final int strideHeight;
  private final int strideWidth;
  private final int kernelHeight;
  private final int kernelWidth;
  private final LayerParameter emptyLayerParam;

  @Inject
  private PoolingLayerParameterInitializer(
      final MatrixFactory matrixFactory,
      @Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
      @Parameter(LayerConfigurationParameters.LayerInputShape.class) final String inputShape,
      @Parameter(LayerConfigurationParameters.StrideHeight.class) final int strideHeight,
      @Parameter(LayerConfigurationParameters.StrideWidth.class) final int strideWidth,
      @Parameter(LayerConfigurationParameters.KernelHeight.class) final int kernelHeight,
      @Parameter(LayerConfigurationParameters.KernelWidth.class) final int kernelWidth) {
    this.index = index;
    this.inputShape = shapeFromString(inputShape);
    this.strideHeight = strideHeight;
    this.strideWidth = strideWidth;
    this.kernelHeight = kernelHeight;
    this.kernelWidth = kernelWidth;
    this.outputShape = computeOutputShape();
    this.emptyLayerParam = LayerParameter.newEmptyInstance(matrixFactory);
  }

  /**
   * @return the initial parameter of the layer.
   */
  public LayerParameter generateInitialParameter() {
    return emptyLayerParam;
  }

  /**
   * @return the index of the layer.
   */
  public int getIndex() {
    return index;
  }

  /**
   * This function checks if stride is set proper for the input shape.
   */
  private void checkShape() {
    if ((inputShape[0] - kernelHeight) % strideHeight != 0) {
      throw new IllegalArgumentException("Stride height is not proper for input.");
    } else if ((inputShape[1] - kernelWidth) % strideWidth != 0) {
      throw new IllegalArgumentException("Stride width is not proper for input.");
    }
  }

  /**
   * This function computes output shape.
   * input shape: row * col
   * output shape: row' * col'
   * row' = (row − kernelHeight) / stride + 1
   * col' = (col − kernelWidth) / stride + 1
   * @return shape of output
   */
  private int[] computeOutputShape() {
    final int[] computedShape = new int[2];
    checkShape();
    computedShape[0] = (inputShape[0] - kernelHeight) / strideHeight + 1;
    computedShape[1] = (inputShape[1] - kernelWidth) / strideWidth + 1;
    return computedShape;
  }

  /**
   * @return shape of output
   */
  @Override
  public int[] getOutputShape() {
    return outputShape;
  }
}
