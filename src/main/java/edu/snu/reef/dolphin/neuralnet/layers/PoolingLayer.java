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
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.NumberOfOutput;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.KernelSize;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.PoolingFunction;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

import com.google.common.base.Preconditions;

/**
 * Pooling Layer.
 * <p/>
 * Implement 1D and 2D max/mean pooling.
 * Pooling is a way of taking the most responsive node of the given interest region.
 * e.g. Max-pooling with kernel_size 2 gets of 4 x 4 input matrix and
 * returns the maximum value for each disjoint sliding window with kernel_size * kernel_size.
 * The derivatives of max-pooling are defined 1 for the node having maximum value and 0 for the others.
 * Mean-pooling works the same way, but returns mean(=average) values for that region.
 * The derivatives of mean-pooling are uniformly 1/(kernel_size * kernel_size) for all.
 * <p/>
 * Reference
 * - http://ufldl.stanford.edu/wiki/index.php/Pooling
 * - http://caffe.berkeleyvision.org/tutorial/layers.html
 */
public final class PoolingLayer implements Layer {

  protected final int index;
  private final int kernelSize;
  private final String poolingFunction;
  private final int numOutput;

  private final INDArray maxDerivative;

  @Inject
  public PoolingLayer(@Parameter(LayerIndex.class) final int index,
                      @Parameter(KernelSize.class) final int kernelSize,
                      @Parameter(PoolingFunction.class) final String poolingFunction,
                      @Parameter(NumberOfOutput.class) final int numOutput) {
    Preconditions.checkArgument(poolingFunction.equalsIgnoreCase("max") || poolingFunction.equalsIgnoreCase("mean"));

    this.index = index;
    this.kernelSize = kernelSize;
    this.poolingFunction = poolingFunction.toLowerCase();
    this.numOutput = numOutput;

    this.maxDerivative = Nd4j.zeros(numOutput * kernelSize * kernelSize);
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public int getIndex() {
    return index;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int getNumOutput() {
    return numOutput;
  }

  /**
   * Do nothing because pooling layer does not have weights and bias.
   */
  @Override
  public void setLayerParameter(final LayerParameter layerParameter) {
  }

  /**
   * Return null because pooling layer does not have weights and bias.
   */
  @Override
  public LayerParameter getLayerParameter() {
    return null;
  }

  /**
   * Return derivatives according to the last activation.
   *
   * @param activation last activation values during feedForward
   */
  @Override
  public INDArray derivative(final INDArray activation) {
    Preconditions.checkArgument(activation.shape().length == 2);
    Preconditions.checkArgument(poolingFunction.equals("max") || poolingFunction.equals("mean"));
    final int[] dim = activation.shape();
    switch (dim[0]) {
    case 1:
      // 1D
      if (poolingFunction.equals("max")) {
        return maxDerivative.subArray(new int[]{0, 0}, new int[]{1, kernelSize * dim[1]}, new int[]{1, 1});
      } else if (poolingFunction.equals("mean")) {
        return Nd4j.valueArrayOf(kernelSize * dim[1], 1.0f / kernelSize);
      }
      break;
    default:
      // 2D
      if (poolingFunction.equals("max")) {
        return maxDerivative.reshape(kernelSize * dim[0], kernelSize * dim[0]);
      } else if (poolingFunction.equals("mean")) {
        return Nd4j.valueArrayOf(kernelSize * dim[0], kernelSize * dim[1],
            1.0f / (kernelSize * kernelSize));
      }
      break;
    }
    throw new IllegalArgumentException("Illegal activation dimensions:" + dim[0]);
  }

  /**
   * Implement max/mean pooling over 2D input.
   *
   * @param input a 1D or 2D matrix with 1 * cols or rows * cols
   * @return a pooled matrix with X/kernelSize or (X/kernelSize * Y/kernelSize) dimension
   */
  @Override
  public INDArray feedForward(final INDArray input) {
    Preconditions.checkArgument(input.shape().length == 2);
    Preconditions.checkArgument(input.shape()[0] == 1 || input.shape()[0] % kernelSize == 0);
    Preconditions.checkArgument(input.shape()[1] % kernelSize == 0);
    Preconditions.checkArgument(poolingFunction.equals("max") || poolingFunction.equals("mean"));

    final int[] inputDim = input.shape();
    final INDArray output = Nd4j.zeros(inputDim[0] == 1 ? new int[]{1, inputDim[1] / kernelSize} :
        new int[]{inputDim[0] / kernelSize, inputDim[1] / kernelSize});

    maxDerivative.assign(0.0f);
    switch (input.shape()[0]) {
    case 1:
      // 1D input
      if (poolingFunction.equals("max")) {
        for (int i = 0; i < inputDim[1]; i += kernelSize) {
          int pos = i;
          for (int k = 0; k < kernelSize; k++) {
            final float num = input.getFloat(0, i + k);
            final float current = output.getFloat(0, i / kernelSize);
            if (num > current) {
              pos = i + k;
              output.putScalar(new int[]{0, i / kernelSize}, num);
            }
          }
          maxDerivative.linearView().putScalar(pos, 1.0f);
        }
      } else if (poolingFunction.equals("mean")) {
        for (int i = 0; i < inputDim[1]; i += kernelSize) {
          float sum = 0.0f;
          for (int k = 0; k < kernelSize; k++) {
            sum += input.getFloat(0, i + k);
          }
          output.putScalar(new int[]{0, i / kernelSize}, sum);
        }
        output.divi(kernelSize);
      }
      break;
    default:
      // 2D input
      if (poolingFunction.equals("max")) {
        for (int i = 0; i < inputDim[0]; i += kernelSize) {
          for (int j = 0; j < inputDim[1]; j += kernelSize) {
            final int[] pos = new int[]{i, j};
            for (int k = 0; k < kernelSize; k++) {
              for (int l = 0; l < kernelSize; l++) {
                final float num = input.getFloat(i + k, j + l);
                final float current = output.getFloat(i / kernelSize, j / kernelSize);
                if (num > current) {
                  pos[0] = i + k;
                  pos[1] = j + l;
                  output.putScalar(new int[]{i / kernelSize, j / kernelSize}, num);
                }
              }
            }
            maxDerivative.reshape(inputDim[0], inputDim[1]).putScalar(new int[]{pos[0], pos[1]}, 1.0f);
          }
        }
      } else if (poolingFunction.equals("mean")) {
        for (int i = 0; i < inputDim[0]; i += kernelSize) {
          for (int j = 0; j < inputDim[1]; j += kernelSize) {
            float sum = 0.0f;
            for (int k = 0; k < kernelSize; k++) {
              for (int l = 0; l < kernelSize; l++) {
                sum += input.getFloat(i + k, j + l);
              }
            }
            output.putScalar(new int[]{i / kernelSize, j / kernelSize}, sum);
          }
        }
        output.divi(kernelSize * kernelSize);
      }
      break;
    }

    return output;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray derivative,
                                final LayerParameter prevParam, final INDArray nextGradient) {
    return nextGradient.mmul(prevParam.getWeightParam().transpose()).muli(derivative);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray label) {
    return activation.sub(label);
  }
}
