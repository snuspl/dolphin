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
import edu.snu.dolphin.dnn.blas.function.Function;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.*;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

/**
 * Pooling layer.
 *
 * This layer is not learnable.
 * This layer resizes input matrix spatially, using max pooling or average pooling.
 * In a forward pass,
 * max pooling picks maximum value in certain range (kernelHeight * kernelWidth) and these values make up output.
 * Average pooling gets average of values in certain range (kernelHeight * kernelWidth) and these values make up output.
 * In a backward pass,
 * each value of error matrix is the sum of elements in next error matrix
 * on which had an impact in feedforward step.
 */
public final class PoolingLayer extends LayerBase {

  private enum PType {
    average, max
  }
  private final int[] outputShape;
  private PType poolingType;
  private int stride;
  private int kernelHeight;
  private int kernelWidth;
  private Matrix trackMatrix;
  private Function poolingFunctions;

  @Inject
  private PoolingLayer(@Parameter(LayerIndex.class) final int index,
                       @Parameter(LayerInputShape.class) final String inputShape,
                       @Parameter(PoolingType.class) final String poolingType,
                       @Parameter(Stride.class) final int stride,
                       @Parameter(KernelHeight.class) final int kernelHeight,
                       @Parameter(KernelWidth.class) final int kernelWidth,
                       final LayerParameterInitializer layerParameterInitializer) {
    super(index, inputShape);
    this.outputShape = setOutputShape();
    this.stride = stride;
    this.kernelHeight = kernelHeight;
    this.kernelWidth = kernelWidth;

    try {
      this.poolingType = PType.valueOf(poolingType);
    } catch (final IllegalArgumentException illegalArgumentException) {
      throw new IllegalArgumentException("Illegal pooling type: " + illegalArgumentException);
    } catch (final NullPointerException nullPointerException) {
      throw new NullPointerException("Null pointer exception while matching pooling type: " + nullPointerException);
    }
  }

  @Override
  public int[] getOutputShape() {
    return outputShape;
  }

  /**
   * This function computes output shape.
   * input shape: row * col
   * output shape: row' * col'
   * row = (row − kernelHeight) / stride + 1
   * col = (col − kernelWidth) / stride + 1
   */
  private int[] setOutputShape() {
    final int[] inputShape = getInputShape();
    final int[] computedShape;
    switch (inputShape.length) {
    case 1 :
      computedShape = new int[1];
      computedShape[0] = (inputShape[0] - kernelHeight) / stride + 1;
      return computedShape;
    case 2 :
      computedShape = new int[2];
      computedShape[0] = (inputShape[0] - kernelHeight) / stride + 1;
      computedShape[1] = (inputShape[1] - kernelWidth) / stride + 1;
      return computedShape;
    default :
      throw new IllegalArgumentException("Unsupported input dimension: " + Integer.toString(inputShape.length));
    }
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return false;
  }

  private Matrix feedForwardMaxPooling(final Matrix input) {
    throw new RuntimeException("Not implemented");
  }

  private Matrix feedForwardAveragePooling(final Matrix input) {
    throw new RuntimeException("Not implemented");
  }

  /**
   * Computes output values for this pooling layer.
   * available pooling type: max, average
   * @param input input values for this layer.
   * @return output values for this layer.
   */
  @Override
  public  Matrix feedForward(final Matrix input) {
    switch (poolingType) {
    case max :
      return feedForwardMaxPooling(input);
    case average :
      return feedForwardAveragePooling(input);
    default :
      throw new IllegalArgumentException("Illegal pooling type: " + poolingType);
    }
  }

  private Matrix backPropagateMaxPooling(final Matrix input, final Matrix nextError) {
    throw new RuntimeException("Not implemented");
  }

  private Matrix backPropagateAveragePooling(final Matrix input, final Matrix nextError) {
    throw new RuntimeException("Not implemented");
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
    case max :
      return backPropagateMaxPooling(input, nextError);
    case average :
      return backPropagateAveragePooling(input, nextError);
    default :
      throw new IllegalArgumentException("Illegal pooling type: " + poolingType);
    }
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final Matrix input, final Matrix error) {
    throw new RuntimeException("This layer doesn't have parameter gradient");
  }
}
