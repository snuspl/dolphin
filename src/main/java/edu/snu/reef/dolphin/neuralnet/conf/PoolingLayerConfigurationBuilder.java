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
package edu.snu.reef.dolphin.neuralnet.conf;

import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.PoolingLayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layers.PoolingLayer;
import edu.snu.reef.dolphin.neuralnet.layers.Layer;
import edu.snu.reef.dolphin.neuralnet.proto.NeuralNetworkProtos;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.util.Builder;

/**
 * Configuration builder for pooling layer.
 * <p/>
 * The configuration that this builder generates is used to create a fully connected layer instance.
 */
public final class PoolingLayerConfigurationBuilder implements Builder<Configuration> {

  public static PoolingLayerConfigurationBuilder newConfigurationBuilder() {
    return new PoolingLayerConfigurationBuilder();
  }

  private int numInput;
  private int numOutput;
  private long randomSeed = System.currentTimeMillis();
  private float initWeight;
  private float initBias;
  private int poolingSize;
  private String poolingFunction;
  private String activationFunction;

  public synchronized PoolingLayerConfigurationBuilder setNumInput(final int numInput) {
    this.numInput = numInput;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setNumOutput(final int numOutput) {
    this.numOutput = numOutput;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setRandomSeed(final long randomSeed) {
    this.randomSeed = randomSeed;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setInitWeight(final float initWeight) {
    this.initWeight = initWeight;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setInitBias(final float initBias) {
    this.initBias = initBias;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setPoolingSize(final int poolingSize) {
    this.poolingSize = poolingSize;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setPoolingFunction(final String poolingFunction) {
    this.poolingFunction = poolingFunction;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setActivationFunction(final String activationFunction) {
    this.activationFunction = activationFunction;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder fromProtoConfiguration(
      final NeuralNetworkProtos.LayerConfiguration protoConf) {
    numInput = protoConf.getNumInput();
    numOutput = protoConf.getNumOutput();

    if (protoConf.getPoolingParam().hasRandomSeed()) {
      randomSeed = protoConf.getPoolingParam().getRandomSeed();
    }
    initWeight = protoConf.getPoolingParam().getInitWeight();
    initBias = protoConf.getPoolingParam().getInitBias();
    poolingSize = protoConf.getPoolingParam().getPoolingSize();
    poolingFunction = protoConf.getPoolingParam().getPoolingFunction();
    activationFunction = protoConf.getPoolingParam().getActivationFunction();
    return this;
  }

  @Override
  public synchronized Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.NumberOfInput.class, String.valueOf(numInput))
        .bindNamedParameter(LayerConfigurationParameters.NumberOfOutput.class, String.valueOf(numOutput))
        .bindNamedParameter(LayerConfigurationParameters.RandomSeed.class, String.valueOf(randomSeed))
        .bindNamedParameter(LayerConfigurationParameters.InitialWeight.class, String.valueOf(initWeight))
        .bindNamedParameter(LayerConfigurationParameters.InitialBias.class, String.valueOf(initBias))
        .bindNamedParameter(LayerConfigurationParameters.PoolingSize.class, String.valueOf(poolingSize))
        .bindNamedParameter(LayerConfigurationParameters.PoolingFunction.class, String.valueOf(poolingFunction))
        .bindNamedParameter(LayerConfigurationParameters.ActivationFunction.class, String.valueOf(activationFunction))
        .bindImplementation(Layer.class, PoolingLayer.class)
        .bindImplementation(LayerParameterInitializer.class, PoolingLayerParameterInitializer.class)
        .build();
  }
}
