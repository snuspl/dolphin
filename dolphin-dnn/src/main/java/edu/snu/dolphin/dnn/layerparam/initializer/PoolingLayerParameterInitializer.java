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
  private final int paddingHeight;
  private final int paddingWidth;
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
      @Parameter(LayerConfigurationParameters.PaddingHeight.class) final int paddingHeight,
      @Parameter(LayerConfigurationParameters.PaddingWidth.class) final int paddingWidth,
      @Parameter(LayerConfigurationParameters.StrideHeight.class) final int strideHeight,
      @Parameter(LayerConfigurationParameters.StrideWidth.class) final int strideWidth,
      @Parameter(LayerConfigurationParameters.KernelHeight.class) final int kernelHeight,
      @Parameter(LayerConfigurationParameters.KernelWidth.class) final int kernelWidth) {
    this.index = index;
    this.inputShape = shapeFromString(inputShape);
    this.paddingHeight = paddingHeight;
    this.paddingWidth = paddingWidth;
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
   * This function computes output shape.
   * input shape: row * col
   * output shape: row' * col'
   * row' = ceil((row − kernelHeight + 2 * paddingHeight) / strideHeight) + 1
   * col' = ceil((col − kernelWidth + 2 * paddingWidth) / strideWidth) + 1
   * @return shape of output
   */
  private int[] computeOutputShape() {
    final int[] computedShape = new int[2];
    if (inputShape.length != 2) {
      throw new IllegalArgumentException("Unsupported input dimensions: " + inputShape.length);
    }
    if (paddingHeight >= kernelHeight) {
      throw new IllegalArgumentException("Padding height should be less than kernel height.");
    }
    if (paddingWidth >= kernelWidth) {
      throw new IllegalArgumentException("Padding width should be less than kernel width.");
    }
    computedShape[0] = (int) Math.ceil((float) (inputShape[0] - kernelHeight + 2 * paddingHeight) / strideHeight) + 1;
    computedShape[1] = (int) Math.ceil((float) (inputShape[1] - kernelWidth + 2 * paddingWidth) / strideWidth) + 1;
    //Pooling should start inside the input images.
    //If the last pooling starts outside the input image, clip that output.
    if ((computedShape[0] - 1) * strideHeight >= inputShape[0] + paddingHeight) {
      --computedShape[0];
      if ((computedShape[0] - 1) * strideHeight >= inputShape[0] + paddingHeight) {
        throw new IllegalArgumentException("The second last pooling still starts outside of the image " +
            "even though we clip the last.");
      }
    }
    if ((computedShape[1] - 1) * strideWidth >= inputShape[1] + paddingWidth) {
      --computedShape[1];
      if ((computedShape[1] - 1) * strideWidth >= inputShape[1] + paddingWidth) {
        throw new IllegalArgumentException("The second last pooling still starts outside of the image " +
            "even though we clip the last.");
      }
    }
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
