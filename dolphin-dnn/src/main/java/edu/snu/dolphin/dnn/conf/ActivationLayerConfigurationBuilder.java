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

import edu.snu.dolphin.dnn.layerparam.initializer.DefaultLayerParameterInitializer;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.ActivationLayer;
import edu.snu.dolphin.dnn.layers.LayerBase;
import edu.snu.dolphin.dnn.proto.NeuralNetworkProtos;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.util.Builder;

/**
 * Configuration builder for activation layer.
 *
 * The configuration that this builder generates is used to create an activation layer instance.
 */
public final class ActivationLayerConfigurationBuilder implements Builder<Configuration> {

  public static ActivationLayerConfigurationBuilder newConfigurationBuilder() {
    return new ActivationLayerConfigurationBuilder();
  }

  private int numInput;
  private int numOutput;
  private String activationFunction;

  public synchronized ActivationLayerConfigurationBuilder setNumInput(final int numInput) {
    this.numInput = numInput;
    return this;
  }

  public synchronized ActivationLayerConfigurationBuilder setNumOutput(final int numOutput) {
    this.numOutput = numOutput;
    return this;
  }

  public synchronized ActivationLayerConfigurationBuilder setActivationFunction(final String activationFunction) {
    this.activationFunction = activationFunction;
    return this;
  }

  public synchronized ActivationLayerConfigurationBuilder fromProtoConfiguration(
      final NeuralNetworkProtos.LayerConfiguration protoConf) {
    numInput = protoConf.getNumInput();
    numOutput = protoConf.getNumOutput();
    activationFunction = protoConf.getActivationParam().getActivationFunction();
    return this;
  }

  @Override
  public synchronized Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.NumberOfInput.class, String.valueOf(numInput))
        .bindNamedParameter(LayerConfigurationParameters.NumberOfOutput.class, String.valueOf(numOutput))
        .bindNamedParameter(LayerConfigurationParameters.ActivationFunction.class, String.valueOf(activationFunction))
        .bindImplementation(LayerBase.class, ActivationLayer.class)
        .bindImplementation(LayerParameterInitializer.class, DefaultLayerParameterInitializer.class)
        .build();
  }
}
