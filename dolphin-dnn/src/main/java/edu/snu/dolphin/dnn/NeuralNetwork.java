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

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.blas.MatrixUtils;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.BatchSize;
import edu.snu.dolphin.dnn.layers.LayerBase;
import edu.snu.dolphin.dnn.layerparam.provider.ParameterProvider;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import javax.inject.Inject;
import java.io.IOException;
import java.util.Set;

/**
 * Neural network model.
 */
public final class NeuralNetwork {

  private final MatrixFactory matrixFactory;

  /**
   * The size of a batch of training inputs.
   * The parameters are only updated after a batch of inputs is processed.
   */
  private final int batchSize;

  /**
   * A set of layers which a neural network comprises.
   */
  private final LayerBase[] layers;

  /**
   * Manager that provides the updated parameters and gathers activations and errors for each input.
   */
  private final ParameterProvider parameterProvider;

  /**
   * The number of processed training inputs.
   */
  private int trainedCount;

  /**
   * the empty matrix.
   * This is used as the next error of the last layer's backpropagation.
   */
  private final Matrix emptyMatrix;
  /**
   * The empty layer parameter.
   * This is used as the gradients of layers that are not learnable.
   */
  private final LayerParameter emptyLayerParam;

  @Inject
  private NeuralNetwork(final MatrixFactory matrixFactory,
                        final ConfigurationSerializer configurationSerializer,
                        @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfSets,
                        @Parameter(BatchSize.class) final int batchSize,
                        final ParameterProvider parameterProvider) {
    this.matrixFactory = matrixFactory;
    this.batchSize = batchSize;
    this.parameterProvider = parameterProvider;
    this.layers = new LayerBase[serializedLayerConfSets.size()];
    this.emptyMatrix = matrixFactory.create(0);
    this.emptyLayerParam = LayerParameter.newEmptyInstance(matrixFactory);

    final Configuration matrixFactoryConf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, matrixFactory.getClass())
        .build();

    for (final String serializedLayerConfiguration : serializedLayerConfSets) {
      try {
        final Configuration layerConfiguration = configurationSerializer.fromString(serializedLayerConfiguration);
        final Injector injector = Tang.Factory.getTang().newInjector(layerConfiguration, matrixFactoryConf);
        final LayerBase layer = injector.getInstance(LayerBase.class);
        this.layers[layer.getIndex()] = layer;

      } catch (final IOException exception) {
        throw new RuntimeException("IOException", exception);
      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException", exception);
      }
    }
  }

  /**
   * @return the parameters of each layer.
   */
  public LayerParameter[] getParameters() {
    final LayerParameter[] parameters = new LayerParameter[layers.length];
    for (int i = 0; i < layers.length; ++i) {
      if (layers[i].isLearnable()) {
        parameters[i] = layers[i].getLayerParameter();
      }
    }
    return parameters;
  }

  /**
   * Trains neural network with the given input and label.
   * @param input the input matrix.
   * @param label the label vector.
   */
  public void train(final Matrix input, final Matrix label) {
    final Matrix[] activations = ArrayUtils.add(feedForward(input), 0, input); // inserts input at the beginning.
    final Matrix[] errors = backPropagate(activations, label);
    final LayerParameter[] parameterGradients = generateParameterGradients(activations, errors);

    parameterProvider.push(parameterGradients);
    
    if (++trainedCount >= batchSize) {
      final LayerParameter[] updatedParameters = parameterProvider.pull();
      for (int i = 0; i < layers.length; ++i) {
        if (layers[i].isLearnable()) {
          layers[i].setLayerParameter(updatedParameters[i]);
        }
      }
      trainedCount = 0;
    }
  }

  /**
   * Trains neural network with the given input and label.
   * @param input the input matrix.
   * @param label the label.
   */
  public void train(final Matrix input, final int label) {
    train(input, MatrixUtils.createOutputVector(matrixFactory, label, layers[layers.length - 1].getNumOutput()));
  }

  /**
   * Computes activations from input layer to output layer.
   * @param input the input matrix for input layer.
   * @return an array of activations for each layer.
   */
  public Matrix[] feedForward(final Matrix input) {
    return feedForward(0, layers.length - 1, input);
  }

  /**
   * Computes activations from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param input the input matrix.
   * @return an array of activations for each layer.
   */
  public Matrix[] feedForward(final int begin, final int end, final Matrix input) {
    if (begin > end) {
      throw new IllegalArgumentException(String.format(
          "The beginning index (%d) must be less than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end, true);

    final Matrix[] activations = new Matrix[end - begin + 1];
    Matrix activation = input;

    for (int i = begin; i <= end; ++i) {
      activation = layers[i].feedForward(activation);
      activations[i - begin] = activation;
    }

    return activations;
  }

  /**
   * Computes errors from the output layer to the input layer.
   * This returns an empty array when the network model has zero or one layer.
   * @param activations an array of activations for each layer.
   * @param label the expected output.
   * @return an array of errors for each layer.
   */
  public Matrix[] backPropagate(final Matrix[] activations,
                                final Matrix label) {
    // Process backpropagation to the second layer.
    // because the error returned by the first layer's backpropagation, is not needed
    // to generate gradients for learning the first layer.
    // The errors for generating gradients used to update the first layer's parameter, are calculated
    // in the next layer's backpropagation.
    return backPropagateTo(1, activations, label);
  }

  /**
   * Computes errors from the output layer to the specified ending layer.
   * This returns an empty array when the network model has zero or one layer.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param label the expected output.
   * @return an array of errors for each layer.
   */
  public Matrix[] backPropagateTo(final int end,
                                  final Matrix[] activations,
                                  final Matrix label) {
    // Case 1: Only one layer
    if (layers.length < 2) {
      // If a neural network has only one layer, the network does not process backpropagation for this layer because
      // this layer cannot generate gradients for updating its parameter.
      // Generating gradients requires the error computed by the next layer.
      return new Matrix[0];
    } else {
      final int lastLayerIndex = layers.length - 1;
      // The first element of activations is the input data.
      // So, (i + 1)-th element of activations refers to the activation of i-th layer.
      final Matrix error = layers[lastLayerIndex].backPropagate(label, activations[lastLayerIndex + 1], emptyMatrix);

      // Case 2: Two layers
      if (lastLayerIndex == end) {
        return new Matrix[]{error};
      // Case 3: More than two layers
      } else {
        return ArrayUtils.add(backPropagateFromTo(lastLayerIndex - 1, end, activations, error), error);
      }
    }
  }

  /**
   * Computes errors from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param nextError the error for next layer - the one closer to the output layer.
   * @return an array of errors for each layer.
   */
  public Matrix[] backPropagateFromTo(final int begin, final int end,
                                      final Matrix[] activations,
                                      final Matrix nextError) {
    if (begin == layers.length - 1) {
      throw new IllegalArgumentException("The beginning layer of backPropagateFromTo cannot be the output layer");
    }
    if (end == 0) {
      // The errors for generating gradients of the first layer are calculated by the next layer, not the first layer.
      throw new IllegalArgumentException("The ending layer cannot be the first layer: " +
          "The error that is propagated to the input is unnecessary to generate gradients for the first layer");
    }
    if (begin < end) {
      throw new IllegalArgumentException(String.format(
          "The beginning index (%d) must be greater than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end, false);

    final Matrix[] errors = new Matrix[begin - end + 1];
    Matrix error = nextError;

    for (int i = begin; i >= end; --i) {
      error = layers[i].backPropagate(activations[i], activations[i + 1], error);
      errors[i - end] = error;
    }
    return errors;
  }

  /**
   * Generates parameter gradients for all layers.
   * @param activations the activation values for each layer.
   * @param errors the errors for each layer.
   * @return an array of parameter gradients for each layer.
   */
  public LayerParameter[] generateParameterGradients(final Matrix[] activations,
                                                     final Matrix[] errors) {
    return generateParameterGradients(0, layers.length - 1, activations, errors);
  }

  /**
   * Generates parameter gradients for each layer from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param activations the activation values for each layer.
   * @param errors the errors for each layer.
   * @return an array of parameter gradients for each layer.
   */
  public LayerParameter[] generateParameterGradients(final int begin, final int end,
                                                     final Matrix[] activations,
                                                     final Matrix[] errors) {
    if (begin > end) {
      throw new IllegalArgumentException(String.format(
          "The beginning index (%d) must be less than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end, true);

    final LayerParameter[] parameterGradients = new LayerParameter[end - begin + 1];
    for (int i = begin; i <= end; ++i) {
      if (layers[i].isLearnable()) {
        parameterGradients[i - begin] = layers[i].generateParameterGradient(activations[i], errors[i]);
      } else {
        parameterGradients[i - begin] = emptyLayerParam;
      }
    }
    return parameterGradients;
  }

  /**
   * Check whether the indices for the beginning layer and the ending layer are within layer bound.
   * @param begin the index of the beginning layer, inclusive.
   * @param end the index of the ending layer, inclusive.
   * @param isForward the flag for a direction.
   */
  private void checkIndices(final int begin, final int end, final boolean isForward) {
    // Case 1: forward direction
    if (isForward) {
      if (begin < 0) {
        throw new IllegalArgumentException(String.format(
            "The beginning index (%d) must be greater than or equal to 0.", begin));
      }
      if (end >= layers.length) {
        throw new IllegalArgumentException(String.format(
            "The ending index (%d) must be less than the length of layers (%d).", end, layers.length));
      }

      // Case 2: backward direction
    } else {
      if (end < 0) {
        throw new IllegalArgumentException(String.format(
            "The ending index (%d) must be greater than or equal to 0.", end));
      }
      if (begin >= layers.length) {
        throw new IllegalArgumentException(String.format(
            "The beginning index (%d) must be less than the length of layers (%d).", begin, layers.length));
      }
    }
  }
}
