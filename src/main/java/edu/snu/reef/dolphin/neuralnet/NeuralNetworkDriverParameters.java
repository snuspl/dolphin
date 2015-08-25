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
import edu.snu.reef.dolphin.neuralnet.conf.FullyConnectedLayerConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.LocalNeuralNetParameterProvider;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.formats.CommandLine;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import javax.inject.Inject;

/**
 * Class that manages command line parameters specific to the neural network for driver.
 */
public final class NeuralNetworkDriverParameters {

  private final String serializedNeuralNetworkConfiguration;
  private final String delimiter;
  private final int maxIterations;

  @NamedParameter(doc = "neural network configuration file path", short_name = "conf")
  public static class ConfigurationPath implements Name<String> {
  }

  @NamedParameter(doc = "delimiter that is used in input file", short_name = "delim", default_value = "   ")
  public static class Delimiter implements Name<String> {
  }

  @Inject
  private NeuralNetworkDriverParameters(final ConfigurationSerializer configurationSerializer,
                                        @Parameter(ConfigurationPath.class) final String configurationPath,
                                        @Parameter(Delimiter.class) final String delimiter,
                                        @Parameter(MaxIterations.class) final int maxIterations) {
    this.serializedNeuralNetworkConfiguration =
        configurationSerializer.toString(loadNeuralNetworkConfiguration(configurationPath));
    this.delimiter = delimiter;
    this.maxIterations = maxIterations;
  }

  /**
   * Loads neural network configuration from the configuration file and parses the configuration.
   * @param path the path for the neural network configuration.
   * @return the neural network configuration.
   */
  private static Configuration loadNeuralNetworkConfiguration(final String path) {
    //TODO #83: read neural network configuration from file.
    return NeuralNetworkConfigurationBuilder.newConfigurationBuilder()
        .setBatchSize(10)
        .setStepSize(1e-2)
        .setParameterProviderClass(LocalNeuralNetParameterProvider.class)
        .addLayerConfiguration(
            FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
                .setNumInput(28 * 28) //MNIST
                .setNumOutput(50)
                .setInitWeight(0.0001)
                .setInitBias(0.0002)
                .setActivationFunction("sigmoid")
                .build())
        .addLayerConfiguration(
            FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
                .setNumInput(50)
                .setNumOutput(10)
                .setInitWeight(0.01)
                .setInitBias(0.02)
                .setActivationFunction("sigmoid")
                .build())
        .build();
  }

  /**
   * Registers command line parameters for driver.
   * @param cl
   */
  public static void registerShortNameOfClass(final CommandLine cl) {
    cl.registerShortNameOfClass(ConfigurationPath.class);
    cl.registerShortNameOfClass(Delimiter.class);
    cl.registerShortNameOfClass(MaxIterations.class);
  }

  /**
   * @return the configuration for driver.
   */
  public Configuration getDriverConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(
            NeuralNetworkTaskParameters.SerializedNeuralNetConf.class,
            serializedNeuralNetworkConfiguration)
        .bindNamedParameter(Delimiter.class, delimiter)
        .bindNamedParameter(MaxIterations.class, String.valueOf(maxIterations))
        .build();
  }
}
