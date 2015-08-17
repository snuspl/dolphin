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

import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterProvider;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.JavaConfigurationBuilder;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.formats.AvroConfigurationSerializer;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.apache.reef.util.Builder;

import java.util.ArrayList;
import java.util.Collection;

public final class NeuralNetworkConfigurationBuilder implements Builder<Configuration> {

  private Collection<String> layerConfigurations = new ArrayList<>();
  private int batchSize = 1;
  private int index = 0;
  private ConfigurationSerializer configurationSerializer = new AvroConfigurationSerializer();
  private Class<? extends ParameterProvider> parameterProviderClass;

  public static NeuralNetworkConfigurationBuilder newConfigurationBuilder() {
    return new NeuralNetworkConfigurationBuilder();
  }

  public NeuralNetworkConfigurationBuilder addLayerConfiguration(final Configuration layerConfiguration) {

    final Configuration finalLayerConfiguration = Configurations.merge(layerConfiguration,
        Tang.Factory.getTang().newConfigurationBuilder()
            .bindNamedParameter(LayerConfigurationParameters.LayerIndex.class, String.valueOf(index))
            .build());

    layerConfigurations.add(configurationSerializer.toString(finalLayerConfiguration));
    ++index;
    return this;
  }

  public NeuralNetworkConfigurationBuilder setBatchSize(final int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public NeuralNetworkConfigurationBuilder setParameterProviderClass(
      final Class<? extends ParameterProvider> parameterProviderClass) {
    this.parameterProviderClass = parameterProviderClass;
    return this;
  }

  @Override
  public Configuration build() {
    final JavaConfigurationBuilder jb = Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(NeuralNetworkParameters.BatchSize.class, String.valueOf(batchSize));

    for (final String layerConfiguration : layerConfigurations) {
      jb.bindSetEntry(NeuralNetworkParameters.SerializedLayerConfigurationSet.class, layerConfiguration);
    }

    jb.bindNamedParameter(NeuralNetworkParameters.ParameterProviderClassName.class, parameterProviderClass.getName());

    return jb.build();
  }
}
