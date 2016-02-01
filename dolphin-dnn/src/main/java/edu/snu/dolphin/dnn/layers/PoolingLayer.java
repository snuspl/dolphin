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
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.*;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

/**
 * Pooling layer.
 *
 * This layer is not learnable.
 * This layer resizes input matrix spatially, using max pooling or average pooling.
 * This layer works for only 1D and 2D inputs.
 * In a forward pass,
 * max pooling picks the maximum value in certain range (kernelHeight * kernelWidth) and these values make up output.
 * Average pooling gets the average of values in certain range (kernelHeight * kernelWidth)
 * and these values make up output.
 * In a backward pass,
 * error of each input pixel comes from errors of output pixels affected by the input pixel in feedforward step.
 */
public final class PoolingLayer extends LayerBase {

  private enum PoolType {
    AVERAGE, MAX
  }
  private final int[] outputShape;
  private final PoolType poolingType;
  private final int strideHeight;
  private final int strideWidth;
  private final int kernelHeight;
  private final int kernelWidth;
  private Matrix indexMatrix;
  private MatrixFactory matrixFactory;

  @Inject
  private PoolingLayer(@Parameter(LayerIndex.class) final int index,
                       @Parameter(LayerInputShape.class) final String inputShape,
                       @Parameter(PoolingType.class) final String poolingType,
                       @Parameter(StrideHeight.class) final int strideHeight,
                       @Parameter(StrideWidth.class) final int strideWidth,
                       @Parameter(KernelHeight.class) final int kernelHeight,
                       @Parameter(KernelWidth.class) final int kernelWidth,
                       final LayerParameterInitializer layerParameterInitializer,
                       final MatrixFactory matrixFactory) {
    super(index, inputShape);
    this.strideHeight = strideHeight;
    this.strideWidth = strideWidth;
    this.kernelHeight = kernelHeight;
    this.kernelWidth = kernelWidth;
    this.outputShape = layerParameterInitializer.getOutputShape();
    this.poolingType = PoolType.valueOf(poolingType);
    this.matrixFactory = matrixFactory;
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
   * Feedforward function for max pooling.
   * @param input the input values for this layer.
   * @return the output values for this layer.
   */
  private Matrix feedForwardMaxPooling(final Matrix input) {
    final int[] inputShape = getInputShape();
    final Matrix output = matrixFactory.create(outputShape[0] * outputShape[1], input.getColumns());
    indexMatrix = matrixFactory.create(outputShape[0] * outputShape[1], input.getColumns());
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix inputImage = input.getColumn(n).reshape(inputShape[0], inputShape[1]).transpose();
      final Matrix outputImage = matrixFactory.create(outputShape[0], outputShape[1]);
      final Matrix indexMatrixImage = matrixFactory.create(outputShape[0], outputShape[1]);
      int ih = 0;
      for (int oh = 0; oh < outputShape[0]; ++oh, ih += strideHeight) {
        int iw = 0;
        for (int ow = 0; ow < outputShape[1]; ++ow, iw += strideWidth) {
          //Find maximum value within kernel range and put it in the output matrix.
          float max = inputImage.get(ih, iw);
          int index = iw + ih * inputImage.getColumns();
          for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {
              final float tempValue = inputImage.get(ih + kh, iw + kw);
              if (tempValue > max) {
                max = tempValue;
                index = iw + kw + (ih + kh) * inputImage.getColumns();
              }
            }
          }
          outputImage.put(oh, ow, max);
          //Save index of max value.
          indexMatrixImage.put(oh, ow, index);
        }
      }
      output.putColumn(n, outputImage.transpose().reshape(outputShape[0] * outputShape[1], 1));
      indexMatrix.putColumn(n, indexMatrixImage.transpose().reshape(outputShape[0] * outputShape[1], 1));
    }
    return output;
  }

  /**
   * Feedforward function for average pooling.
   * @param input the input values for this layer.
   * @return the output values for this layer.
   */
  private Matrix feedForwardAveragePooling(final Matrix input) {
    final int[] inputShape = getInputShape();
    final int kernelSize = kernelHeight * kernelWidth;
    final Matrix output = matrixFactory.create(outputShape[0] * outputShape[1], input.getColumns());
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix inputImage = input.getColumn(n).reshape(inputShape[0], inputShape[1]).transpose();
      final Matrix outputImage = matrixFactory.create(outputShape[0], outputShape[1]);
      int ih = 0;
      for (int oh = 0; oh < outputShape[0]; ++oh, ih += strideHeight) {
        int iw = 0;
        for (int ow = 0; ow < outputShape[1]; ++ow, iw += strideWidth) {
          //Find sum of values within kernel range and put average value in the output matrix.
          float sum = 0;
          for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {
              sum += inputImage.get(ih + kh, iw + kw);
            }
          }
          outputImage.put(oh, ow, sum / kernelSize);
        }
      }
      output.putColumn(n, outputImage.transpose().reshape(outputShape[0] * outputShape[1], 1));
    }
    return output;
  }

  /**
   * Computes output values for this pooling layer.
   * available pooling type: max, average
   * @param input the input values for this layer.
   * @return the output values for this layer.
   */
  @Override
  public Matrix feedForward(final Matrix input) {
    switch (poolingType) {
    case MAX:
      return feedForwardMaxPooling(input);
    case AVERAGE:
      return feedForwardAveragePooling(input);
    default:
      throw new IllegalArgumentException("Illegal pooling type: " + poolingType);
    }
  }

  /**
   * Backpropagating function for max pooling.
   * @param input the input values for this layer.
   * @param nextError the errors of the next layer - the one closer to the output layer.
   * @return errors for this layer with the specified input value.
   */
  private Matrix backPropagateMaxPooling(final Matrix input, final Matrix nextError) {
    final Matrix error = matrixFactory.zeros(input.getRows(), input.getColumns());
    final int[] inputShape = getInputShape();
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix errorImage = matrixFactory.zeros(inputShape[0], inputShape[1]);
      final Matrix nextErrorImage = nextError.getColumn(n).reshape(outputShape[0], outputShape[1]).transpose();
      for (int oh = 0; oh < outputShape[0]; ++oh) {
        for (int ow = 0; ow < outputShape[1]; ++ow) {
          final int ih = (int) indexMatrix.get(oh * outputShape[1] + ow, n) / errorImage.getColumns();
          final int iw = (int) indexMatrix.get(oh * outputShape[1] + ow, n) - ih * errorImage.getColumns();
          final float tempError = nextErrorImage.get(oh, ow) + errorImage.get(ih, iw);
          //Add error to saved index.
          errorImage.put(ih, iw, tempError);
        }
      }
      error.putColumn(n, errorImage.transpose().reshape(inputShape[0] * inputShape[1], 1));
    }
    return error;
  }

  /**
   * Backpropagating function for average pooling.
   * @param input the input values for this layer.
   * @param nextError the errors of the next layer - the one closer to the output layer.
   * @return errors for this layer with the specified input value.
   */
  private Matrix backPropagateAveragePooling(final Matrix input, final Matrix nextError) {
    final float kernelSize = kernelHeight * kernelWidth;
    final Matrix error = matrixFactory.zeros(input.getRows(), input.getColumns());
    final int[] inputShape = getInputShape();
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix errorImage = matrixFactory.zeros(inputShape[0], inputShape[1]);
      final Matrix nextErrorImage = nextError.getColumn(n).reshape(outputShape[0], outputShape[1]).transpose();
      for (int oh = 0; oh < outputShape[0]; ++oh) {
        for (int ow = 0; ow < outputShape[1]; ++ow) {
          final int sh = strideHeight * oh;
          final int sw = strideWidth * ow;
          for (int ih = sh; ih < sh + kernelHeight; ++ih) {
            for (int iw = sw; iw < sw + kernelWidth; ++iw) {
              //Add error divided by kernel size for all pixels within the range.
              final float tempError = nextErrorImage.get(oh, ow) / kernelSize + errorImage.get(ih, iw);
              errorImage.put(ih, iw, tempError);
            }
          }
        }
      }
      error.putColumn(n, errorImage.transpose().reshape(inputShape[0] * inputShape[1], 1));
    }
    return error;
  }

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
    switch (poolingType) {
    case MAX:
      return backPropagateMaxPooling(input, nextError);
    case AVERAGE:
      return backPropagateAveragePooling(input, nextError);
    default:
      throw new IllegalArgumentException("Illegal pooling type: " + poolingType);
    }
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final Matrix input, final Matrix error) {
    throw new RuntimeException("This layer is not learnable");
  }
}
