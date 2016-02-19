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
import edu.snu.dolphin.dnn.util.NeuralNetworkUtils;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

/**
 * Convolutional layer.
 *
 * This layer is learnable having the updatable parameter (weight and bias).
 * This layer works for only 2D inputs.
 * In a forward pass,
 * feedForward function computes the product between weight and the input within kernel range
 * and produce activation matrix.
 * In a backward pass,
 * the error of each input pixel comes from the product
 * between weight and errors of output pixels affected by the input pixel in feedforward step.
 */
public final class ConvolutionalLayer extends LayerBase {

  private final int[] outputShape;
  private final int paddingHeight;
  private final int paddingWidth;
  private final int strideHeight;
  private final int strideWidth;
  private final int kernelHeight;
  private final int kernelWidth;
  private final MatrixFactory matrixFactory;

  @Inject
  private ConvolutionalLayer(@Parameter(LayerIndex.class) final int index,
                             @Parameter(LayerInputShape.class) final String inputShape,
                             @Parameter(PaddingHeight.class) final int paddingHeight,
                             @Parameter(PaddingWidth.class) final int paddingWidth,
                             @Parameter(StrideHeight.class) final int strideHeight,
                             @Parameter(StrideWidth.class) final int strideWidth,
                             @Parameter(KernelHeight.class) final int kernelHeight,
                             @Parameter(KernelWidth.class) final int kernelWidth,
                             final LayerParameterInitializer layerParameterInitializer,
                             final MatrixFactory matrixFactory) {
    super(index, inputShape);
    this.paddingHeight = paddingHeight;
    this.paddingWidth = paddingWidth;
    this.strideHeight = strideHeight;
    this.strideWidth = strideWidth;
    this.kernelHeight = kernelHeight;
    this.kernelWidth = kernelWidth;
    this.outputShape = layerParameterInitializer.getOutputShape();
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
    this.matrixFactory = matrixFactory;
  }

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
   * Transform the given image to column form to facilitate matrix multiplication.
   * @param imageIndex the index of the image in the input matrix.
   * @param input input values for this layer.
   * @return the converted column.
   */
  private Matrix im2col(final int imageIndex, final Matrix input) {
    final int[] inputShape = getInputShape();
    final Matrix col =
        matrixFactory.zeros(kernelHeight * kernelWidth, NeuralNetworkUtils.getShapeLength(outputShape));
    for (int kh = 0; kh < kernelHeight; ++kh) {
      for (int kw = 0; kw < kernelWidth; ++kw) {
        int ih = kh - paddingHeight;
        for (int oh = 0; oh < outputShape[0]; ++oh) {
          if (ih >= 0 && ih < inputShape[0]) {
            int iw = kw - paddingWidth;
            for (int ow = 0; ow < outputShape[1]; ++ow) {
              if (iw >= 0 && iw < inputShape[1]) {
                col.put(kh * kernelWidth + kw, oh * outputShape[1] + ow,
                    input.get(ih * inputShape[1] + iw, imageIndex));
              }
              iw += strideWidth;
            }
          }
          ih += strideHeight;
        }
      }
    }
    return col;
  }

  /**
   * Transform the given column form to the image to facilitate matrix multiplication.
   * @param col the given column.
   * @return the converted image.
   */
  private Matrix col2im(final Matrix col) {
    final int[] inputShape = getInputShape();
    final int kernelSize = kernelHeight * kernelWidth;
    final Matrix im = matrixFactory.zeros(NeuralNetworkUtils.getShapeLength(inputShape), 1);
    int colIndex = 0;
    for (int kh = 0; kh < kernelHeight; ++kh) {
      for (int kw = 0; kw < kernelWidth; ++kw) {
        int ih = kh - paddingHeight;
        for (int oh = 0; oh < outputShape[0]; ++oh) {
          if (ih < 0 || ih >= inputShape[0]) {
            colIndex += outputShape[1];
          } else {
            int iw = kw - paddingWidth;
            for (int ow = 0; ow < outputShape[1]; ++ow) {
              if (iw >= 0 && iw < inputShape[1]) {
                final int colh = colIndex / kernelSize;
                final int colw = colIndex - colh * kernelSize;
                final int inputIndex = ih * inputShape[1] + iw;
                final float newValue = col.get(colh, colw) + im.get(inputIndex);
                im.put(inputIndex, newValue);
              }
              colIndex++;
              iw += strideWidth;
            }
          }
          ih += strideHeight;
        }
      }
    }
    return im;
  }

  /**
   * Computes output values for this convolutional layer.
   * @param input input values for this layer.
   * @return output values for this layer.
   */
  @Override
  public Matrix feedForward(final Matrix input) {
    final Matrix output = matrixFactory.create(NeuralNetworkUtils.getShapeLength(outputShape), input.getColumns());
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix col = im2col(n, input);
      output.putColumn(n, getLayerParameter().getWeightParam().transpose().mmul(col));
    }
    output.addiColumnVector(getLayerParameter().getBiasParam());
    return output;
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
    final Matrix error = matrixFactory.create(input.getRows(), input.getColumns());
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix im = col2im(nextError.getColumn(n).mmul(getLayerParameter().getWeightParam().transpose()));
      error.putColumn(n, im);
    }
    return error;
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final Matrix input, final Matrix error) {
    final Matrix weightGradient = matrixFactory.create(kernelHeight * kernelWidth, 1);
    for (int n = 0; n < input.getColumns(); ++n) {
      final Matrix col = im2col(n, input);
      weightGradient.addiColumnVector(col.mmul(error.getColumn(n)));
    }
    return LayerParameter.newBuilder()
        .setWeightParam(weightGradient)
        .setBiasParam(error.rowSums())
        .build();
  }
}
