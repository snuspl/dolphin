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

import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.BatchSize;
import edu.snu.reef.dolphin.neuralnet.layers.LayerBase;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;

import javax.inject.Inject;
import java.io.IOException;
import java.util.Set;

/**
 * Neural network model.
 */
public final class NeuralNetwork {

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

  @Inject
  private NeuralNetwork(final ConfigurationSerializer configurationSerializer,
                        @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfSets,
                        @Parameter(BatchSize.class) final int batchSize,
                        final ParameterProvider parameterProvider) {
    this.batchSize = batchSize;
    this.parameterProvider = parameterProvider;
    this.layers = new LayerBase[serializedLayerConfSets.size()];

    for (final String serializedLayerConfiguration : serializedLayerConfSets) {
      try {
        final Configuration layerConfiguration = configurationSerializer.fromString(serializedLayerConfiguration);
        final Injector injector = Tang.Factory.getTang().newInjector(layerConfiguration);
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
  public void train(final INDArray input, final INDArray label) {
    final INDArray[] activations = ArrayUtils.add(feedForward(input), 0, input); // inserts input at the beginning.
    final INDArray[] errors = backPropagate(activations, label);
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
  public void train(final INDArray input, final int label) {
    train(input, FeatureUtil.toOutcomeVector(label, layers[layers.length - 1].getNumOutput()));
  }

  /**
   * Computes activations from input layer to output layer.
   * @param input the input matrix for input layer.
   * @return an array of activations for each layer.
   */
  public INDArray[] feedForward(final INDArray input) {
    return feedForward(0, layers.length - 1, input);
  }

  /**
   * Computes activations from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param input the input matrix.
   * @return an array of activations for each layer.
   */
  public INDArray[] feedForward(final int begin, final int end, final INDArray input) {
    if (begin > end) {
      throw new RuntimeException(String.format(
          "The beginning index (%d) must be less than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end);

    final INDArray[] activations = new INDArray[end - begin + 1];
    INDArray activation = input;

    for (int i = begin; i <= end; ++i) {
      activation = layers[i].feedForward(activation);
      activations[i - begin] = activation;
    }

    return activations;
  }

  /**
   * Computes errors from output layer to input layer.
   * @param activations an array of activations for each layer.
   * @param label the expected output.
   * @return an array of errors for each layer.
   */
  public INDArray[] backPropagate(final INDArray[] activations,
                                  final INDArray label) {
    return backPropagateTo(0, activations, label);
  }

  /**
   * Computes errors from output layer to the specified ending layer.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param label the expected output.
   * @return an array of errors for each layer.
   */
  public INDArray[] backPropagateTo(final int end,
                                    final INDArray[] activations,
                                    final INDArray label) {
    final int lastLayerIndex = layers.length - 1;

    // The first element of activations is input data.
    // So, (i + 1)-th element of activations refers to the activation of i-th layer.
    final INDArray error = layers[lastLayerIndex].backPropagate(activations[lastLayerIndex + 1], label);
    return ArrayUtils.add(backPropagateFromTo(lastLayerIndex - 1, end, activations, error), error);
  }

  /**
   * Computes errors from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param nextError the error for next layer - the one closer to the output layer.
   * @return an array of errors for each layer.
   */
  public INDArray[] backPropagateFromTo(final int begin, final int end,
                                        final INDArray[] activations,
                                        final INDArray nextError) {
    if (begin == layers.length - 1) {
      throw new RuntimeException("The beginning layer of backPropagateFromTo cannot be output layer");
    }
    if (begin < end) {
      throw new RuntimeException(String.format(
          "The beginning index (%d) must be greater than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end);

    final INDArray[] errors = new INDArray[end - begin + 1];
    INDArray error = nextError;

    for (int i = begin; i >= end; --i) {
      error = layers[i].backPropagate(activations[i + 1], layers[i + 1].getLayerParameter(), error);
      errors[i - begin] = error;
    }
    return errors;
  }

  /**
   * Generates parameter gradients for all layers.
   * @param activations activation values for each layer.
   * @param errors errors for each layer.
   * @return an array of parameter gradients for each layer.
   */
  public LayerParameter[] generateParameterGradients(final INDArray[] activations,
                                                     final INDArray[] errors) {
    return generateParameterGradients(0, layers.length - 1, activations, errors);
  }

  /**
   * Generates parameter gradients for each layer from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param activations activation values for each layer.
   * @param errors errors for each layer.
   * @return an array of parameter gradients for each layer.
   */
  public LayerParameter[] generateParameterGradients(final int begin, final int end,
                                                     final INDArray[] activations,
                                                     final INDArray[] errors) {
    if (begin > end) {
      throw new RuntimeException(String.format(
          "The beginning index (%d) must be less than or equal to the ending index (%d).", begin, end));
    }

    checkIndices(begin, end);

    final LayerParameter[] parameterGradients = new LayerParameter[end - begin + 1];
    for (int i = begin; i <= end; ++i) {
      parameterGradients[i - begin] = layers[i].generateParameterGradient(activations[i], errors[i]);
    }
    return parameterGradients;
  }

  /**
   * Check whether the indices for the beginning layer and the ending layer are within layer bound.
   * @param begin the index of the beginning layer, inclusive.
   * @param end the index of the ending layer, inclusive.
   */
  private void checkIndices(final int begin, final int end) {
    // Case 1: forward direction
    if (begin < end) {
      if (begin < 0) {
        throw new RuntimeException(String.format(
            "The beginning index (%d) must be greater than or equal to 0.", begin));
      }
      if (end >= layers.length) {
        throw new RuntimeException(String.format(
            "The ending index (%d) must be less than the length of layers (%d).", end, layers.length));
      }

      // Case 2: backward direction
    } else {
      if (end < 0) {
        throw new RuntimeException(String.format("The ending index (%d) must be greater than or equal to 0.", end));
      }
      if (begin >= layers.length) {
        throw new RuntimeException(String.format(
            "The beginning index (%d) must be less than the length of layers (%d).", begin, layers.length));
      }
    }
  }
}
