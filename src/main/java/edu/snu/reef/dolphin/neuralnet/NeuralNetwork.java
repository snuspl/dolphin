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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
   * Manager that provides the updated parameters and gathers activations and error gradient vector for each input.
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
    final List<INDArray> activations = new ArrayList<>();
    activations.add(input);
    activations.addAll(feedForward(input));

    final List<INDArray> errorGradients = backPropagate(activations, label);

    parameterProvider.push(activations, errorGradients);
    
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
   * @return a list of activations for each layer.
   */
  public List<INDArray> feedForward(final INDArray input) {
    return feedForward(0, layers.length - 1, input);
  }

  /**
   * Computes activations from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param input the input matrix.
   * @return a list of activations for each layer.
   */
  public List<INDArray> feedForward(final int begin, final int end, final INDArray input) {
    final List<INDArray> activations = new ArrayList<>();

    INDArray activation = input;
    for (int i = begin; i <= end; ++i) {
      activation = layers[i].feedForward(activation);
      activations.add(activation);
    }

    return activations;
  }

  /**
   * Computes error gradients from output layer to input layer.
   * @param activations an array of activations for each layer.
   * @param label the expected output.
   * @return a list of error gradients for each layer.
   */
  public List<INDArray> backPropagate(final List<INDArray> activations,
                                      final INDArray label) {
    return backPropagateTo(0, activations, label);
  }

  /**
   * Computes error gradients from output layer to the specified ending layer.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param label the expected output.
   * @return a list of error gradients for each layer.
   */
  public List<INDArray> backPropagateTo(final int end,
                                        final List<INDArray> activations,
                                        final INDArray label) {
    final int lastLayerIndex = layers.length - 1;
    final INDArray errorGradient = layers[lastLayerIndex].backPropagate(activations.get(lastLayerIndex + 1), label);
    final List<INDArray> errorGradients = backPropagateFromTo(lastLayerIndex - 1, end, activations, errorGradient);
    errorGradients.add(errorGradient);
    return errorGradients;
  }

  /**
   * Computes error gradients from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer, inclusive.
   * @param end the index of ending layer, inclusive.
   * @param activations an array of activations of each layer.
   * @param nextErrorGradient the error gradient for next layer.
   * @return a list of error gradients for each layer.
   */
  public List<INDArray> backPropagateFromTo(final int begin, final int end,
                                            final List<INDArray> activations,
                                            final INDArray nextErrorGradient) {
    if (begin == layers.length - 1) {
      throw new RuntimeException("The beginning layer of backPropagateFromTo cannot be output layer");
    }
    if (begin < end) {
      throw new RuntimeException("The beginning index must be greater than or equal to the ending index.");
    }

    final List<INDArray> errorGradients = new ArrayList<>();
    INDArray errorGradient = nextErrorGradient;

    for (int i = begin; i >= end; --i) {
      errorGradient = layers[i].backPropagate(activations.get(i + 1), layers[i + 1].getLayerParameter(), errorGradient);
      errorGradients.add(errorGradient);
    }

    Collections.reverse(errorGradients);
    return errorGradients;
  }
}
