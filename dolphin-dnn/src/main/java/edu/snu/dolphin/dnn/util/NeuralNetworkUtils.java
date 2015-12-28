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
package edu.snu.dolphin.dnn.util;

import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.LayerBase;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Utility class for neural network.
 */
public final class NeuralNetworkUtils {

  private NeuralNetworkUtils() {
  }

  /**
   * Delimiter that is used for distinguishing dimensions of shapes.
   */
  private static final String SHAPE_DELIMITER = ",";

  /**
   * Converts a list of integers for a shape to a string.
   * @param dimensionList a list of integers for a shape.
   * @return a string for a shape.
   */
  public static String shapeToString(final List<Integer> dimensionList) {
    return StringUtils.join(dimensionList, SHAPE_DELIMITER);
  }

  /**
   * Converts an array of {@code int}s for a shape to a string.
   * @param dimensions an array of {@code int}s for a shape.
   * @return a string for a shape.
   */
  public static String shapeToString(final int[] dimensions) {
    return shapeToString(Arrays.asList(ArrayUtils.toObject(dimensions)));
  }

  /**
   * Converts a string for a shape to an array of integers.
   * @param shapeString a string for a shape.
   * @return an array of integers for a shape.
   */
  public static int[] shapeFromString(final String shapeString) {
    final String[] inputShapeStrings = shapeString.split(SHAPE_DELIMITER);
    final int[] inputShape = new int[inputShapeStrings.length];
    for (int i = 0; i < inputShapeStrings.length; ++i) {
      inputShape[i] = Integer.parseInt(inputShapeStrings[i]);
    }
    return inputShape;
  }

  public static int getShapeLength(final int[] shape) {
    if (shape.length < 1) {
      throw new IllegalArgumentException("the shape must have one or more dimensions");
    }

    int length = shape[0];
    for (int i = 1; i < shape.length; ++i) {
      length *= shape[i];
    }

    if (length > 0) {
      return length;
    } else {
      throw new IllegalArgumentException("the length of the shape must be positive: " + length);
    }
  }

  /**
   * De-serializes a set of serialized layer configurations to an array of layer configurations.
   * @param configurationSerializer a configuration serializer to deserialize the specified configuration set.
   * @param serializedLayerConfSet a set of serialized layer configurations.
   * @return an array of layer configurations.
   */
  public static Configuration[] deserializeLayerConfSetToArray(
      final ConfigurationSerializer configurationSerializer,
      final Set<String> serializedLayerConfSet) {
    final Configuration[] layerConfigurations = new Configuration[serializedLayerConfSet.size()];
    for (final String serializedLayerConfiguration : serializedLayerConfSet) {
      try {
        final Configuration layerConfiguration = configurationSerializer.fromString(serializedLayerConfiguration);
        final int index = Tang.Factory.getTang().newInjector(layerConfiguration)
            .getNamedInstance(LayerConfigurationParameters.LayerIndex.class);
        layerConfigurations[index] = layerConfiguration;
      } catch (final IOException exception) {
        throw new RuntimeException("IOException while de-serializing layer configuration", exception);
      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException", exception);
      }
    }
    return layerConfigurations;
  }

  /**
   * Returns initial layer parameters from the specified set of configurations and the input shape.
   * @param layerInitializerConfs an array of configurations for injecting layer initializers
   * @param inputShape an input shape for the neural network.
   * @param matrixFactoryClass a matrix factory class for instantiate matrices
   * @return an array of initial layer parameters
   */
  public static LayerParameter[] getInitialLayerParameters(final Configuration[] layerInitializerConfs,
                                                           final String inputShape,
                                                           final Class<? extends MatrixFactory> matrixFactoryClass) {
    final LayerParameter[] layerParameters = new LayerParameter[layerInitializerConfs.length];

    String currentInputShape = inputShape;
    for (int i = 0; i < layerInitializerConfs.length; ++i) {
      try {
        // bind an input shape for the layer and the matrix factory.
        final Configuration finalInitializerConf =
            Tang.Factory.getTang().newConfigurationBuilder(layerInitializerConfs[i])
                .bindNamedParameter(LayerConfigurationParameters.LayerInputShape.class, currentInputShape)
                .bindImplementation(MatrixFactory.class, matrixFactoryClass)
                .build();
        final LayerParameterInitializer layerParameterInitializer =
            Tang.Factory.getTang().newInjector(finalInitializerConf).getInstance(LayerParameterInitializer.class);

        layerParameters[i] = layerParameterInitializer.generateInitialParameter();
        currentInputShape = shapeToString(layerParameterInitializer.getOutputShape());

      } catch (final InjectionException e) {
        throw new RuntimeException("InjectionException while injecting LayerParameterInitializer", e);
      }
    }
    return layerParameters;
  }

  /**
   * Returns initial layer parameters using the specified injector and the specified array of configurations.
   * This assumes that the specified injector has all parameters that are needed to inject layer parameter initializer
   * instances except the configuration for each layer.
   * @param injector an injector used for injecting layer parameter initializer instances
   * @param layerInitializerConfs an array of configurations for injecting layer parameter initializer instances
   * @param inputShape an input shape for the neural network.
   * @return an array of initial layer parameters
   */
  public static LayerParameter[] getInitialLayerParameters(final Injector injector,
                                                           final Configuration[] layerInitializerConfs,
                                                           final String inputShape) {
    final LayerParameter[] layerParameters = new LayerParameter[layerInitializerConfs.length];

    String currentInputShape = inputShape;
    for (int i = 0; i < layerInitializerConfs.length; ++i) {
      try {
        // bind an input shape for the layer.
        final Configuration finalInitializerConf =
            Tang.Factory.getTang().newConfigurationBuilder(layerInitializerConfs[i])
                .bindNamedParameter(LayerConfigurationParameters.LayerInputShape.class, currentInputShape)
                .build();
        final LayerParameterInitializer layerParameterInitializer =
            injector.forkInjector(finalInitializerConf).getInstance(LayerParameterInitializer.class);
        final int index = layerParameterInitializer.getIndex();

        layerParameters[index] = layerParameterInitializer.generateInitialParameter();
        currentInputShape = shapeToString(layerParameterInitializer.getOutputShape());

      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException during injecting LayerParameterInitializer", exception);
      }
    }
    return layerParameters;
  }

  /**
   * Returns layer instances from the specified array of configurations for layer instances.
   * @param layerConfs an array of configurations for layer instances
   * @param inputShape an input shape for the neural network
   * @param matrixFactoryClass a class for {@link MatrixFactory}
   * @return an array of layer instances
   */
  public static LayerBase[] getLayerInstances(final Configuration[] layerConfs,
                                              final String inputShape,
                                              final Class<? extends MatrixFactory> matrixFactoryClass) {
    final LayerBase[] layers = new LayerBase[layerConfs.length];

    String currentInputShape = inputShape;
    for (int i = 0; i < layerConfs.length; ++i) {
      try {
        // bind an input shape for the layer and the matrix factory.
        final Configuration finalLayerConf = Tang.Factory.getTang().newConfigurationBuilder(layerConfs[i])
            .bindNamedParameter(LayerConfigurationParameters.LayerInputShape.class, currentInputShape)
            .bindImplementation(MatrixFactory.class, matrixFactoryClass)
            .build();
        final Injector injector = Tang.Factory.getTang().newInjector(finalLayerConf);
        final LayerBase layer = injector.getInstance(LayerBase.class);

        layers[i] = layer;
        currentInputShape = shapeToString(layer.getOutputShape());

      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException while injecting LayerBase", exception);
      }
    }
    return layers;
  }

  /**
   * Returns layer instances using the specified injector and the specified array of configurations.
   * This assumes that the specified injector has all parameters that are needed to inject layer instances
   * except the configuration for each layer.
   * @param injector an injector used for injecting layer instances
   * @param layerConfs an array of configurations for injecting layer instances
   * @param inputShape an input shape for the neural network.
   * @return an array of layer instances.
   */
  public static LayerBase[] getLayerInstances(final Injector injector,
                                              final Configuration[] layerConfs,
                                              final String inputShape) {
    final LayerBase[] layers = new LayerBase[layerConfs.length];

    String currentInputShape = inputShape;
    for (int i = 0; i < layerConfs.length; ++i) {
      try {
        // bind an input shape for the layer.
        final Configuration finalLayerConf = Tang.Factory.getTang().newConfigurationBuilder(layerConfs[i])
            .bindNamedParameter(LayerConfigurationParameters.LayerInputShape.class, currentInputShape)
            .build();
        final LayerBase layer = injector.forkInjector(finalLayerConf).getInstance(LayerBase.class);

        layers[layer.getIndex()] = layer;
        currentInputShape = shapeToString(layer.getOutputShape());

      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException while injecting LayerBase", exception);
      }
    }
    return layers;
  }
}
