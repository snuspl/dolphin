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

import edu.snu.dolphin.dnn.layerparam.provider.ParameterProvider;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.*;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.JavaConfigurationBuilder;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.formats.AvroConfigurationSerializer;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.apache.reef.util.Builder;

import java.util.ArrayList;
import java.util.List;

import static edu.snu.dolphin.dnn.util.NeuralNetworkUtils.shapeToString;

/**
 * Configuration builder for neural network.
 *
 * The configuration that this builder generates is used to create a neural network instance.
 * The generated configuration should be merged with the BLAS configuration.
 */
public final class NeuralNetworkConfigurationBuilder implements Builder<Configuration> {

  private List<Configuration> layerConfigurations = new ArrayList<>();
  private ConfigurationSerializer configurationSerializer = new AvroConfigurationSerializer();
  private Class<? extends ParameterProvider> parameterProviderClass;
  private float stepsize = 1e-2f;
  private String inputShape;

  public static NeuralNetworkConfigurationBuilder newConfigurationBuilder() {
    return new NeuralNetworkConfigurationBuilder();
  }

  public synchronized NeuralNetworkConfigurationBuilder addLayerConfiguration(final Configuration layerConfiguration) {
    layerConfigurations.add(layerConfiguration);
    return this;
  }

  public synchronized NeuralNetworkConfigurationBuilder setParameterProviderClass(
      final Class<? extends ParameterProvider> parameterProviderClass) {
    this.parameterProviderClass = parameterProviderClass;
    return this;
  }

  public synchronized NeuralNetworkConfigurationBuilder setStepsize(final float stepsize) {
    this.stepsize = stepsize;
    return this;
  }

  public synchronized NeuralNetworkConfigurationBuilder setInputShape(final List<Integer> inputShapeList) {
    this.inputShape = shapeToString(inputShapeList);
    return this;
  }

  public synchronized NeuralNetworkConfigurationBuilder setInputShape(final int... inputShape) {
    this.inputShape = shapeToString(inputShape);
    return this;
  }

  @Override
  public synchronized Configuration build() {
    final JavaConfigurationBuilder jb = Tang.Factory.getTang().newConfigurationBuilder();

    for (int i = 0; i < layerConfigurations.size(); ++i) {
      final Configuration finalLayerConfiguration =
          Tang.Factory.getTang().newConfigurationBuilder(layerConfigurations.get(i))
              .bindNamedParameter(LayerConfigurationParameters.LayerIndex.class, String.valueOf(i))
              .build();

      jb.bindSetEntry(SerializedLayerConfigurationSet.class,
          configurationSerializer.toString(finalLayerConfiguration));
    }

    jb.bindImplementation(ParameterProvider.class, parameterProviderClass);
    jb.bindNamedParameter(Stepsize.class, String.valueOf(stepsize));
    jb.bindNamedParameter(InputShape.class, inputShape);

    return jb.build();
  }
}
