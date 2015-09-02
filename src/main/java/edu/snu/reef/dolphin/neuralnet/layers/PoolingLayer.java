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
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.PoolingSize;
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
 */
public final class PoolingLayer implements Layer {

  protected final int index;
  private final int poolingSize;
  private final String poolingFunction;
  private final int numOutput;

  private final INDArray maxDerivative;

  @Inject
  public PoolingLayer(@Parameter(LayerIndex.class) final int index,
                      @Parameter(PoolingSize.class) final int poolingSize,
                      @Parameter(PoolingFunction.class) final String poolingFunction,
                      @Parameter(NumberOfOutput.class) final int numOutput) {
    Preconditions.checkArgument(poolingFunction.equalsIgnoreCase("max") || poolingFunction.equalsIgnoreCase("mean"));

    this.index = index;
    this.poolingSize = poolingSize;
    this.poolingFunction = poolingFunction.toLowerCase();
    this.numOutput = numOutput;

    this.maxDerivative = Nd4j.zeros(numOutput * poolingSize * poolingSize);
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
        return maxDerivative.subArray(new int[]{0, 0}, new int[]{1, poolingSize * dim[1]}, new int[]{1, 1});
      } else if (poolingFunction.equals("mean")) {
        return Nd4j.valueArrayOf(poolingSize * dim[1], 1.0f / poolingSize);
      }
      break;
    default:
      // 2D
      if (poolingFunction.equals("max")) {
        return maxDerivative.reshape(poolingSize * dim[0], poolingSize * dim[0]);
      } else if (poolingFunction.equals("mean")) {
        return Nd4j.valueArrayOf(poolingSize * dim[0], poolingSize * dim[1],
            1.0f / (poolingSize * poolingSize));
      }
      break;
    }
    throw new IllegalArgumentException("Illegal activation dimensions:" + dim[0]);
  }

  /**
   * Implement max/mean pooling over 2D input.
   *
   * @param input a 1D or 2D matrix with 1 * cols or rows * cols
   * @return a pooled matrix with X/poolingSize or (X/poolingSize * Y/poolingSize) dimension
   */
  @Override
  public INDArray feedForward(final INDArray input) {
    Preconditions.checkArgument(input.shape().length == 2);
    Preconditions.checkArgument(input.shape()[0] == 1 || input.shape()[0] % poolingSize == 0);
    Preconditions.checkArgument(input.shape()[1] % poolingSize == 0);
    Preconditions.checkArgument(poolingFunction.equals("max") || poolingFunction.equals("mean"));

    final int[] inputDim = input.shape();
    final INDArray output = Nd4j.zeros(inputDim[0] == 1 ? new int[]{1, inputDim[1] / poolingSize} :
        new int[]{inputDim[0] / poolingSize, inputDim[1] / poolingSize});

    maxDerivative.assign(0.0f);
    switch (input.shape()[0]) {
    case 1:
      // 1D input
      if (poolingFunction.equals("max")) {
        for (int i = 0; i < inputDim[1]; i += poolingSize) {
          int pos = i;
          for (int k = 0; k < poolingSize; k++) {
            final float num = input.getFloat(0, i + k);
            final float current = output.getFloat(0, i / poolingSize);
            if (num > current) {
              pos = i + k;
              output.putScalar(new int[]{0, i / poolingSize}, num);
            }
          }
          maxDerivative.linearView().putScalar(pos, 1.0f);
        }
      } else if (poolingFunction.equals("mean")) {
        for (int i = 0; i < inputDim[1]; i += poolingSize) {
          float sum = 0.0f;
          for (int k = 0; k < poolingSize; k++) {
            sum += input.getFloat(0, i + k);
          }
          output.putScalar(new int[]{0, i / poolingSize}, sum);
        }
        output.divi(poolingSize);
      }
      break;
    default:
      // 2D input
      if (poolingFunction.equals("max")) {
        for (int i = 0; i < inputDim[0]; i += poolingSize) {
          for (int j = 0; j < inputDim[1]; j += poolingSize) {
            final int[] pos = new int[]{i, j};
            for (int k = 0; k < poolingSize; k++) {
              for (int l = 0; l < poolingSize; l++) {
                final float num = input.getFloat(i + k, j + l);
                final float current = output.getFloat(i / poolingSize, j / poolingSize);
                if (num > current) {
                  pos[0] = i + k;
                  pos[1] = j + l;
                  output.putScalar(new int[]{i / poolingSize, j / poolingSize}, num);
                }
              }
            }
            maxDerivative.reshape(inputDim[0], inputDim[1]).putScalar(new int[]{pos[0], pos[1]}, 1.0f);
          }
        }
      } else if (poolingFunction.equals("mean")) {
        for (int i = 0; i < inputDim[0]; i += poolingSize) {
          for (int j = 0; j < inputDim[1]; j += poolingSize) {
            float sum = 0.0f;
            for (int k = 0; k < poolingSize; k++) {
              for (int l = 0; l < poolingSize; l++) {
                sum += input.getFloat(i + k, j + l);
              }
            }
            output.putScalar(new int[]{i / poolingSize, j / poolingSize}, sum);
          }
        }
        output.divi(poolingSize * poolingSize);
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
