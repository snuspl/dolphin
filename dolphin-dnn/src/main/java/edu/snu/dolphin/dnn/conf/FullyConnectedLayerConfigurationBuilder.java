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
package edu.snu.dolphin.dnn.conf;

import edu.snu.dolphin.dnn.layerparam.initializer.FullyConnectedLayerParameterInitializer;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.FullyConnectedLayer;
import edu.snu.dolphin.dnn.layers.LayerBase;
import edu.snu.dolphin.dnn.proto.NeuralNetworkProtos;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.util.Builder;

/**
 * Configuration builder for fully connected layer.
 *
 * The configuration that this builder generates is used to create a fully connected layer instance.
 * The generated configuration needs to bind the implementation for matrix factory and
 * the parameter for a layer input shape, to inject a layer instance.
 */
public final class FullyConnectedLayerConfigurationBuilder implements Builder<Configuration> {

  public static FullyConnectedLayerConfigurationBuilder newConfigurationBuilder() {
    return new FullyConnectedLayerConfigurationBuilder();
  }

  private int numOutput;
  private long randomSeed = System.currentTimeMillis();
  private float initWeight;
  private float initBias;

  public synchronized FullyConnectedLayerConfigurationBuilder setNumOutput(final int numOutput) {
    this.numOutput = numOutput;
    return this;
  }

  public synchronized FullyConnectedLayerConfigurationBuilder setRandomSeed(final long randomSeed) {
    this.randomSeed = randomSeed;
    return this;
  }

  public synchronized FullyConnectedLayerConfigurationBuilder setInitWeight(final float initWeight) {
    this.initWeight = initWeight;
    return this;
  }

  public synchronized FullyConnectedLayerConfigurationBuilder setInitBias(final float initBias) {
    this.initBias = initBias;
    return this;
  }

  public synchronized FullyConnectedLayerConfigurationBuilder fromProtoConfiguration(
      final NeuralNetworkProtos.LayerConfiguration protoConf) {
    numOutput = protoConf.getFullyConnectedParam().getNumOutput();
    if (protoConf.getFullyConnectedParam().hasRandomSeed()) {
      randomSeed = protoConf.getFullyConnectedParam().getRandomSeed();
    }
    initWeight = protoConf.getFullyConnectedParam().getInitWeight();
    initBias = protoConf.getFullyConnectedParam().getInitBias();
    return this;
  }

  @Override
  public synchronized Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.NumberOfOutput.class, String.valueOf(numOutput))
        .bindNamedParameter(LayerConfigurationParameters.RandomSeed.class, String.valueOf(randomSeed))
        .bindNamedParameter(LayerConfigurationParameters.InitialWeight.class, String.valueOf(initWeight))
        .bindNamedParameter(LayerConfigurationParameters.InitialBias.class, String.valueOf(initBias))
        .bindImplementation(LayerBase.class, FullyConnectedLayer.class)
        .bindImplementation(LayerParameterInitializer.class, FullyConnectedLayerParameterInitializer.class)
        .build();
  }
}
