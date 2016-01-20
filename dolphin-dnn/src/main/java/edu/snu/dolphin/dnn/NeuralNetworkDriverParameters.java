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
package edu.snu.dolphin.dnn;

import com.google.protobuf.TextFormat;
import edu.snu.dolphin.bsp.examples.ml.parameters.MaxIterations;
import edu.snu.dolphin.dnn.NeuralNetworkParameterUpdater.LogPeriod;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.conf.*;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.BatchSize;
import edu.snu.dolphin.dnn.layerparam.provider.GroupCommParameterProvider;
import edu.snu.dolphin.dnn.layerparam.provider.LocalNeuralNetParameterProvider;
import edu.snu.dolphin.dnn.layerparam.provider.ParameterProvider;
import edu.snu.dolphin.dnn.layerparam.provider.ParameterServerParameterProvider;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.dolphin.dnn.proto.NeuralNetworkProtos.*;
import edu.snu.dolphin.bsp.parameters.OnLocal;
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

/**
 * Class that manages command line parameters specific to the neural network for driver.
 */
public final class NeuralNetworkDriverParameters {

  private final String serializedNeuralNetworkConfiguration;
  private final String delimiter;
  private final int maxIterations;
  private final ProviderType providerType;
  private final int logPeriod;
  private final String serializedBlasConfiguration;
  private final int batchSize;

  @NamedParameter(doc = "neural network configuration file path", short_name = "conf")
  public static final class ConfigurationPath implements Name<String> {
  }

  @NamedParameter(doc = "delimiter that is used in input file", short_name = "delim", default_value = ",")
  public static class Delimiter implements Name<String> {
  }

  @NamedParameter(doc = "backend BLAS library", short_name = "blas", default_value = "jblas")
  public static class BlasLibrary implements Name<String> {
  }

  enum ProviderType {
    LOCAL, GROUP_COMM, PARAMETER_SERVER
  }

  @Inject
  private NeuralNetworkDriverParameters(final ConfigurationSerializer configurationSerializer,
                                        @Parameter(ConfigurationPath.class) final String configurationPath,
                                        @Parameter(Delimiter.class) final String delimiter,
                                        @Parameter(MaxIterations.class) final int maxIterations,
                                        @Parameter(OnLocal.class) final boolean onLocal,
                                        @Parameter(LogPeriod.class) final int logPeriod,
                                        @Parameter(BlasLibrary.class) final String blasLibrary) throws IOException {
    final NeuralNetworkConfiguration neuralNetConf = loadNeuralNetworkConfiguration(configurationPath, onLocal);

    // the method is being called twice: here and in `buildNeuralNetworkConfiguration`
    // this could be made to once by refactoring the code
    this.providerType = getProviderType(neuralNetConf.getParameterProvider());
    this.serializedNeuralNetworkConfiguration = configurationSerializer.toString(
        buildNeuralNetworkConfiguration(neuralNetConf));
    this.delimiter = delimiter;
    this.maxIterations = maxIterations;
    this.logPeriod = logPeriod;
    this.serializedBlasConfiguration = configurationSerializer.toString(buildBlasConfiguration(blasLibrary));
    this.batchSize = neuralNetConf.getBatchSize();
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
   * @param blasLibraryType a BLAS library string
   * @return the matrix factory class related to the specified BLAS library string
   */
  private static Class<? extends MatrixFactory> getMatrixFactoryClass(final String blasLibraryType) {
    switch (blasLibraryType.toLowerCase()) {
    case "jblas":
      return MatrixJBLASFactory.class;
    default:
      throw new IllegalArgumentException("Unsupported BLAS library: " + blasLibraryType);
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
    case "activation":
      return ActivationLayerConfigurationBuilder.newConfigurationBuilder()
          .fromProtoConfiguration(layerConf).build();
    case "activationwithloss":
      return ActivationWithLossLayerConfigurationBuilder.newConfigurationBuilder()
          .fromProtoConfiguration(layerConf).build();
    case "pooling":
      return PoolingLayerConfigurationBuilder.newConfigurationBuilder()
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

    neuralNetConfBuilder.setStepsize(neuralNetConf.getStepsize())
        .setParameterProviderClass(getParameterProviderClass(neuralNetConf.getParameterProvider().getType()))
        .setInputShape(neuralNetConf.getInputShape().getDimList());

    // Adds the configuration of each layer.
    for (final LayerConfiguration layerConf : neuralNetConf.getLayerList()) {
      neuralNetConfBuilder.addLayerConfiguration(createLayerConfiguration(layerConf));
    }

    return neuralNetConfBuilder.build();
  }

  /**
   * @param blasLibrary a string that indicates a BLAS library to be used
   * @return the configuration for BLAS library
   */
  private static Configuration buildBlasConfiguration(final String blasLibrary) {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, getMatrixFactoryClass(blasLibrary))
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
    cl.registerShortNameOfClass(LogPeriod.class);
    cl.registerShortNameOfClass(BlasLibrary.class);
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
        .bindNamedParameter(LogPeriod.class, String.valueOf(logPeriod))
        .bindNamedParameter(NeuralNetworkESParameters.SerializedBlasConf.class, serializedBlasConfiguration)
        .bindNamedParameter(BatchSize.class, String.valueOf(batchSize))
        .build();
  }

  /**
   * @return the {@link ProviderType} of the neural net configuration loaded from the filesystem
   */
  public ProviderType getProviderType() {
    return this.providerType;
  }
}
