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
package edu.snu.reef.dolphin.neuralnet.layerparam.initializer;

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Parameter Initializer of pooling connected layer.
 * <p/>
 * initializes the weight matrix with pseudo random normal distributed value with mean 0 and given standard deviation.
 * initializes the bias vector with the given value.
 */
public final class PoolingLayerParameterInitializer implements LayerParameterInitializer {

  private final int index;
  private final int numInput;
  private final int numOutput;
  private final float initWeight;
  private final float initBias;
  private final long randomSeed;
  private final int poolingSize;
  private final String poolingFunc;

  @Inject
  public PoolingLayerParameterInitializer(
      @Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
      @Parameter(LayerConfigurationParameters.NumberOfInput.class) final int numInput,
      @Parameter(LayerConfigurationParameters.NumberOfOutput.class) final int numOutput,
      @Parameter(LayerConfigurationParameters.RandomSeed.class) final long randomSeed,
      @Parameter(LayerConfigurationParameters.InitialWeight.class) final float initWeight,
      @Parameter(LayerConfigurationParameters.InitialBias.class) final float initBias,
      @Parameter(LayerConfigurationParameters.PoolingSize.class) final int poolingSize,
      @Parameter(LayerConfigurationParameters.PoolingFunction.class) final String poolingFunc) {
    this.index = index;
    this.randomSeed = randomSeed;
    this.numInput = numInput;
    this.numOutput = numOutput;
    this.initWeight = initWeight;
    this.initBias = initBias;
    this.poolingSize = poolingSize;
    this.poolingFunc = poolingFunc;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public LayerParameter generateInitialParameter() {
    final INDArray weight = Nd4j.randn(numInput, numOutput, randomSeed);
    final INDArray bias = Nd4j.valueArrayOf(1, numOutput, initBias);

    weight.muli(initWeight); // multiply by standard deviation.

    // Mark arrays persist for the aggressive garbage collection strategy.
    weight.data().persist();
    bias.data().persist();

    return LayerParameter.newBuilder()
        .setWeightParam(weight)
        .setBiasParam(bias)
        .build();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int getIndex() {
    return this.index;
  }
}
