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

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.ActivationFunction;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.LayerIndex;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.NumberOfOutput;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.PoolingSize;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.PoolingFunction;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Pooling Layer.
 */
public final class PoolingLayer implements Layer {

  protected final int index;
  private final int poolingSize;
  private final String poolingFunction;
  private final String activationFunction;
  private final int numOutput;
  private LayerParameter layerParameter;

  private final INDArray maxDerivative;

  @Inject
  public PoolingLayer(@Parameter(LayerIndex.class) final int index,
                      @Parameter(PoolingSize.class) final int poolingSize,
                      @Parameter(PoolingFunction.class) final String poolingFunction,
                      @Parameter(ActivationFunction.class) final String activationFunction,
                      @Parameter(NumberOfOutput.class) final int numOutput,
                      final LayerParameterInitializer layerParameterInitializer) {
    this.index = index;
    this.poolingSize = poolingSize;
    this.poolingFunction = poolingFunction;
    this.activationFunction = activationFunction;
    this.numOutput = numOutput;
    setLayerParameter(layerParameterInitializer.generateInitialParameter());

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
   * {@inheritDoc}
   */
  @Override
  public void setLayerParameter(final LayerParameter layerParameter) {
    this.layerParameter = layerParameter;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public LayerParameter getLayerParameter() {
    return this.layerParameter;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public INDArray derivative(final INDArray activation) {
    final INDArray derivative = Nd4j.zeros(poolingSize * activation.shape()[0], poolingSize * activation.shape()[1]);
    if (poolingFunction.equals("max")) {
      return maxDerivative.reshape(derivative.shape()[0], derivative.shape()[1]);
    } else if (poolingFunction.equals("mean")) {
      derivative.addi(1.0f / (poolingSize * poolingSize));
    }
    return derivative;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public INDArray feedForward(final INDArray input) {
    final int[] inputDim = input.shape();
    final int[] outputDim = new int[]{inputDim[0] / poolingSize, inputDim[1] / poolingSize};
    final INDArray output = Nd4j.zeros(outputDim);
    maxDerivative.assign(0.0f);

    if (poolingFunction.equals("max")) {
      for (int i = 0; i < inputDim[0]; i += poolingSize) {
        for (int j = 0; j < inputDim[1]; j += poolingSize) {
          final int[] pos = new int[] {i, j};
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
          maxDerivative.reshape(inputDim[0], inputDim[1]).putScalar(new int[] {pos[0], pos[1]}, 1.0f);
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
