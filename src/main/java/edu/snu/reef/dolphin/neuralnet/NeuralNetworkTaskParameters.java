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

import org.apache.reef.tang.Configuration;
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

  @NamedParameter(doc = "serialized neural network configuration")
  public static class SerializedNeuralNetworkConfiguration implements Name<String> {
  }

  @Inject
  private NeuralNetworkTaskParameters(final ConfigurationSerializer configurationSerializer,
                                      @Parameter(SerializedNeuralNetworkConfiguration.class)
                                          final String serializedNeuralNetworkConfiguration) throws IOException {
    this.neuralNetworkConfiguration = configurationSerializer.fromString(serializedNeuralNetworkConfiguration);
  }

  /**
   * @return the configuration for task.
   */
  public Configuration getTaskConfiguration() {
    return neuralNetworkConfiguration;
  }
}
