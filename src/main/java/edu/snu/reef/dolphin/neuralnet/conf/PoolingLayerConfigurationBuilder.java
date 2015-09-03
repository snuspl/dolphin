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
 * The configuration that this builder generates is used to create a pooling layer instance.
 */
public final class PoolingLayerConfigurationBuilder implements Builder<Configuration> {

  public static PoolingLayerConfigurationBuilder newConfigurationBuilder() {
    return new PoolingLayerConfigurationBuilder();
  }

  private int numInput;
  private int numOutput;
  private int kernelSize;
  private String poolingFunction;

  public synchronized PoolingLayerConfigurationBuilder setNumInput(final int numInput) {
    this.numInput = numInput;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setNumOutput(final int numOutput) {
    this.numOutput = numOutput;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setKernelSize(final int kernelSize) {
    this.kernelSize = kernelSize;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setPoolingFunction(final String poolingFunction) {
    this.poolingFunction = poolingFunction;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder fromProtoConfiguration(
      final NeuralNetworkProtos.LayerConfiguration protoConf) {
    numInput = protoConf.getNumInput();
    numOutput = protoConf.getNumOutput();
    kernelSize = protoConf.getPoolingParam().getKernelSize();
    poolingFunction = protoConf.getPoolingParam().getPoolingFunction();
    return this;
  }

  @Override
  public synchronized Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.NumberOfInput.class, String.valueOf(numInput))
        .bindNamedParameter(LayerConfigurationParameters.NumberOfOutput.class, String.valueOf(numOutput))
        .bindNamedParameter(LayerConfigurationParameters.KernelSize.class, String.valueOf(kernelSize))
        .bindNamedParameter(LayerConfigurationParameters.PoolingFunction.class, String.valueOf(poolingFunction))
        .bindImplementation(Layer.class, PoolingLayer.class)
        .bindImplementation(LayerParameterInitializer.class, PoolingLayerParameterInitializer.class)
        .build();
  }
}
