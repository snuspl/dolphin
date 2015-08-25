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
package edu.snu.reef.dolphin.neuralnet;

import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import javax.inject.Inject;
import java.io.IOException;

/**
 * Class that manages parameters specific to the neural network for task.
 */
public final class NeuralNetworkTaskParameters {

  private final Configuration neuralNetworkConfiguration;
  private final String delimiter;
  private final int maxIterations;

  @NamedParameter(doc = "serialized neural network configuration")
  public static class SerializedNeuralNetConf implements Name<String> {
  }

  @Inject
  private NeuralNetworkTaskParameters(final ConfigurationSerializer configurationSerializer,
                                      @Parameter(SerializedNeuralNetConf.class) final String serializedNeuralNetConf,
                                      @Parameter(NeuralNetworkDriverParameters.Delimiter.class) final String delimiter,
                                      @Parameter(MaxIterations.class) final int maxIterations) throws IOException {
    this.neuralNetworkConfiguration = configurationSerializer.fromString(serializedNeuralNetConf);
    this.delimiter = delimiter;
    this.maxIterations = maxIterations;
  }

  /**
   * @return the configuration for task.
   */
  public Configuration getTaskConfiguration() {
    return Configurations.merge(Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(NeuralNetworkDriverParameters.Delimiter.class, delimiter)
        .bindNamedParameter(MaxIterations.class, String.valueOf(maxIterations))
        .build(),
        neuralNetworkConfiguration);
  }
}
