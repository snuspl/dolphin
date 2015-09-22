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

import com.google.protobuf.TextFormat;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.neuralnet.NeuralNetworkParameterUpdater.LogPeriod;
import edu.snu.reef.dolphin.neuralnet.conf.FullyConnectedLayerConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.GroupCommParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.LocalNeuralNetParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterServerParameterProvider;
import edu.snu.reef.dolphin.neuralnet.proto.NeuralNetworkProtos.*;
import edu.snu.reef.dolphin.parameters.OnLocal;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.formats.CommandLine;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import javax.inject.Inject;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

/**
 * Class that manages command line parameters specific to the neural network for driver.
 */
public final class NeuralNetworkDriverParameters {

  private final String serializedNeuralNetworkConfiguration;
  private final String delimiter;
  private final int maxIterations;
  private final ProviderType providerType;
  private final String inputShape;
  private final int logPeriod;

  @NamedParameter(doc = "neural network configuration file path", short_name = "conf")
  public static class ConfigurationPath implements Name<String> {
  }

  @NamedParameter(doc = "delimiter that is used in input file", short_name = "delim", default_value = ",")
  public static class Delimiter implements Name<String> {
  }

  @NamedParameter(doc = "the shape of input data")
  public static class InputShape implements Name<String> {
  }

  enum ProviderType {
    LOCAL, GROUP_COMM, PARAMETER_SERVER
  }

  /**
   * Delimiter that is used for distinguishing dimensions of input shape.
   */
  private static final String SHAPE_DELIMITER = ",";

  /**
   * Converts a list of integer for an input shape to a string.
   * @param dimensionList a list of integers for an input shape.
   * @return a string for an input shape.
   */
  public static String inputShapeToString(final List<Integer> dimensionList) {
    return StringUtils.join(dimensionList, SHAPE_DELIMITER);
  }

  /**
   * Converts a string for an input shape to an array of integers.
   * @param inputShapeString a string for an input shape.
   * @return an array of integers for an input shape.
   */
  public static int[] inputShapeFromString(final String inputShapeString) {
    final String[] inputShapeStrings = inputShapeString.split(SHAPE_DELIMITER);
    final int[] inputShape = new int[inputShapeStrings.length];
    for (int i = 0; i < inputShapeStrings.length; ++i) {
      inputShape[i] = Integer.parseInt(inputShapeStrings[i]);
    }
    return inputShape;
  }

  @Inject
  private NeuralNetworkDriverParameters(final ConfigurationSerializer configurationSerializer,
                                        @Parameter(ConfigurationPath.class) final String configurationPath,
                                        @Parameter(Delimiter.class) final String delimiter,
                                        @Parameter(MaxIterations.class) final int maxIterations,
                                        @Parameter(OnLocal.class) final boolean onLocal,
                                        @Parameter(LogPeriod.class) final int logPeriod) throws IOException {
    final NeuralNetworkConfiguration neuralNetConf = loadNeuralNetworkConfiguration(configurationPath, onLocal);

    // the method is being called twice: here and in `buildNeuralNetworkConfiguration`
    // this could be made to once by refactoring the code
    this.providerType = getProviderType(neuralNetConf.getParameterProvider());
    this.serializedNeuralNetworkConfiguration = configurationSerializer.toString(
        buildNeuralNetworkConfiguration(neuralNetConf));
    this.delimiter = delimiter;
    this.maxIterations = maxIterations;
    this.logPeriod = logPeriod;

    // convert to string because Tang configuration serializer does not support List serialization.
    this.inputShape = inputShapeToString(neuralNetConf.getInputShape().getDimList());
  }

  /**
   * @param parameterProviderConfiguration parameter provider protobuf configuration of a neural network
   * @return the type of the configured parameter provider
   */
  private static ProviderType getProviderType(final ParameterProviderConfiguration parameterProviderConfiguration) {
    switch (parameterProviderConfiguration.getType().toLowerCase()) {
    case "local":
      return ProviderType.LOCAL;
    case "groupcomm":
      return ProviderType.GROUP_COMM;
    case "parameterserver":
      return ProviderType.PARAMETER_SERVER;
    default:
      throw new IllegalArgumentException("Illegal parameter provider: " + parameterProviderConfiguration.getType());
    }
  }

  /**
   * @param parameterProvider a parameter provider string.
   * @return the parameter provider class that the given string indicates.
   */
  private static Class<? extends ParameterProvider> getParameterProviderClass(final String parameterProvider) {
    switch (parameterProvider.toLowerCase()) {
    case "local":
      return LocalNeuralNetParameterProvider.class;
    case "groupcomm":
      return GroupCommParameterProvider.class;
    case "parameterserver":
      return ParameterServerParameterProvider.class;
    default:
      throw new IllegalArgumentException("Illegal parameter provider: " + parameterProvider);
    }
  }

  /**
   * Creates the layer configuration from the given protocol buffer layer configuration message.
   * @param layerConf the protocol buffer layer configuration message.
   * @return the layer configuration built from the protocol buffer layer configuration message.
   */
  private static Configuration createLayerConfiguration(final LayerConfiguration layerConf) {
    switch (layerConf.getType().toLowerCase()) {
    case "fullyconnected":
      return FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
          .fromProtoConfiguration(layerConf).build();
    default:
      throw new IllegalArgumentException("Illegal layer type: " + layerConf.getType());
    }
  }

  /**
   * Loads the protocol buffer text formatted neural network configuration.
   * <p/>
   * Loads the file from the local filesystem or HDFS depending on {@code onLocal}.
   * @param path the path for the neural network configuration.
   * @param onLocal the flag for the local runtime environment.
   * @return the neural network configuration protocol buffer message.
   * @throws IOException
   */
  private static NeuralNetworkConfiguration loadNeuralNetworkConfiguration(final String path, final boolean onLocal)
      throws IOException {
    final NeuralNetworkConfiguration.Builder neuralNetProtoBuilder = NeuralNetworkConfiguration.newBuilder();

    // Parses neural network builder protobuf message from the prototxt file.
    // Reads from the local filesystem.
    if (onLocal) {
      TextFormat.merge(new FileReader(path), neuralNetProtoBuilder);
    // Reads from HDFS.
    } else {
      final FileSystem fs = FileSystem.get(new JobConf());
      TextFormat.merge(new InputStreamReader(fs.open(new Path(path))), neuralNetProtoBuilder);
    }
    return neuralNetProtoBuilder.build();
  }

  /**
   * Parses the protobuf message and builds neural network configuration.
   * @param neuralNetConf neural network configuration protobuf message.
   * @return the neural network configuration.
   */
  private static Configuration buildNeuralNetworkConfiguration(final NeuralNetworkConfiguration neuralNetConf) {
    final NeuralNetworkConfigurationBuilder neuralNetConfBuilder =
        NeuralNetworkConfigurationBuilder.newConfigurationBuilder();

    neuralNetConfBuilder.setBatchSize(neuralNetConf.getBatchSize())
        .setStepsize(neuralNetConf.getStepsize())
        .setParameterProviderClass(getParameterProviderClass(neuralNetConf.getParameterProvider().getType()));

    // Adds the configuration of each layer.
    for (final LayerConfiguration layerConf : neuralNetConf.getLayerList()) {
      neuralNetConfBuilder.addLayerConfiguration(createLayerConfiguration(layerConf));
    }

    return neuralNetConfBuilder.build();
  }

  /**
   * Registers command line parameters for driver.
   * @param cl
   */
  public static void registerShortNameOfClass(final CommandLine cl) {
    cl.registerShortNameOfClass(ConfigurationPath.class);
    cl.registerShortNameOfClass(Delimiter.class);
    cl.registerShortNameOfClass(MaxIterations.class);
    cl.registerShortNameOfClass(LogPeriod.class);
  }

  /**
   * @return the configuration for driver.
   */
  public Configuration getDriverConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(
            NeuralNetworkESParameters.SerializedNeuralNetConf.class,
            serializedNeuralNetworkConfiguration)
        .bindNamedParameter(Delimiter.class, delimiter)
        .bindNamedParameter(MaxIterations.class, String.valueOf(maxIterations))
        .bindNamedParameter(InputShape.class, inputShape)
        .bindNamedParameter(LogPeriod.class, String.valueOf(logPeriod))
        .build();
  }

  /**
   * @return {@code true} if this Neural Network application uses REEF Group Communication, {@code false} otherwise
   */
  public ProviderType getProviderType() {
    return this.providerType;
  }
}
