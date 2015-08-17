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

import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.FullyConnectedLayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layers.FullyConnectedLayer;
import edu.snu.reef.dolphin.neuralnet.layers.Layer;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.util.Builder;

public final class FullyConnectedLayerConfigurationBuilder implements Builder<Configuration> {

  public static FullyConnectedLayerConfigurationBuilder newConfigurationBuilder() {
    return new FullyConnectedLayerConfigurationBuilder();
  }

  private int numInput;
  private int numOutput;
  private double initWeight;
  private double initBias;

  public FullyConnectedLayerConfigurationBuilder setNumInput(final int numInput) {
    this.numInput = numInput;
    return this;
  }

  public FullyConnectedLayerConfigurationBuilder setNumOutput(final int numOutput) {
    this.numOutput = numOutput;
    return this;
  }

  public FullyConnectedLayerConfigurationBuilder setInitWeight(final double initWeight) {
    this.initWeight = initWeight;
    return this;
  }

  public FullyConnectedLayerConfigurationBuilder setInitBias(final double initBias) {
    this.initBias = initBias;
    return this;
  }

  @Override
  public Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.NumberOfInput.class, String.valueOf(numInput))
        .bindNamedParameter(LayerConfigurationParameters.NumberOfOutput.class, String.valueOf(numOutput))
        .bindNamedParameter(LayerConfigurationParameters.InitialWeight.class, String.valueOf(initWeight))
        .bindNamedParameter(LayerConfigurationParameters.InitialBias.class, String.valueOf(initBias))
        .bindImplementation(Layer.class, FullyConnectedLayer.class)
        .bindImplementation(LayerParameterInitializer.class, FullyConnectedLayerParameterInitializer.class)
        .build();
  }
}
