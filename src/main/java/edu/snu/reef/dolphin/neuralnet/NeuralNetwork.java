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
import edu.snu.reef.dolphin.neuralnet.layers.Layer;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.io.network.util.Pair;
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

  private final int batchSize;
  private final Layer[] layers;
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
    this.layers = new Layer[serializedLayerConfSets.size()];

    for (final String serializedLayerConfiguration : serializedLayerConfSets) {
      try {
        final Configuration layerConfiguration = configurationSerializer.fromString(serializedLayerConfiguration);
        final Injector injector = Tang.Factory.getTang().newInjector(layerConfiguration);
        final Layer layer = injector.getInstance(Layer.class);
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
      parameters[i] = layers[i].getLayerParameter();
    }
    return parameters;
  }

  /**
   * Trains neural network with the given input and label.
   * @param input the input matrix.
   * @param label the label vector.
   */
  public void train(final INDArray input, final INDArray label) {
    final Pair<List<INDArray>, List<INDArray>> actAndDeriv = activationAndDerivative(input);
    final List<INDArray> activations = actAndDeriv.getFirst();
    final List<INDArray> derivatives = actAndDeriv.getSecond();

    final List<INDArray> gradients = backPropagate(activations, derivatives, label);

    parameterProvider.push(activations, gradients);
    
    if (++trainedCount >= batchSize) {
      final LayerParameter[] updatedParameters = parameterProvider.pull();
      for (int i = 0; i < layers.length; ++i) {
        layers[i].setLayerParameter(updatedParameters[i]);
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
    return feedForward(0, layers.length, input);
  }

  /**
   * Computes activations from the specified beginning layer to the specified ending layer.
   * @param begin the index of beginning layer.
   * @param end the index of ending layer.
   * @param input the input matrix.
   * @return a list of activations for each layer.
   */
  public List<INDArray> feedForward(final int begin, final int end, final INDArray input) {
    final List<INDArray> activations = new ArrayList<>();

    INDArray activation = input;
    activations.add(activation);

    for (int i = begin; i < end; ++i) {
      activation = layers[i].feedForward(activation);
      activations.add(activation);
    }

    return activations;
  }

  /**
   * Computes activations and derivatives from input layer to output layer.
   * @param input the input matrix.
   * @return a pair of a list of activations for each layer and a list of derivatives for each layer.
   */
  public Pair<List<INDArray>, List<INDArray>> activationAndDerivative(final INDArray input) {
    return activationAndDerivative(0, layers.length, input);
  }

  /**
   * Computes activations and derivatives from the specified beginning layer and the specified ending layer.
   * @param begin the index of beginning layer.
   * @param end the index of ending layer.
   * @param input the input matrix for beginning layer.
   * @return a pair of a list of activations for each layer and a list of derivatives for each layer.
   */
  public Pair<List<INDArray>, List<INDArray>> activationAndDerivative(final int begin,
                                                                      final int end,
                                                                      final INDArray input) {
    final List<INDArray> activations = new ArrayList<>();
    final List<INDArray> derivatives = new ArrayList<>();

    INDArray activation = input;
    activations.add(activation);
    derivatives.add(activation); // Dummy derivative to match index of activation and derivative lists.

    for (int i = begin; i < end; ++i) {
      activation = layers[i].feedForward(activation);
      activations.add(activation);
      derivatives.add(layers[i].derivative(activation));
    }

    return new Pair<>(activations, derivatives);
  }

  /**
   * Computes gradients from output layer to input layer.
   * @param activations a list of activations for each layer.
   * @param derivatives a list of derivatives for each layer.
   * @param label the expected output.
   * @return a list of gradients for each layer.
   */
  public List<INDArray> backPropagate(final List<INDArray> activations,
                                      final List<INDArray> derivatives,
                                      final INDArray label) {
    return backPropagateTo(0, activations, derivatives, label);
  }

  /**
   * Computes gradients from output layer and the specified ending layer.
   * @param to the index of ending layer.
   * @param activations a list of activations of each layer.
   * @param derivatives a list of derivatives of each layer.
   * @param label the expected output.
   * @return a list of gradients for each layer.
   */
  public List<INDArray> backPropagateTo(final int to,
                                        final List<INDArray> activations,
                                        final List<INDArray> derivatives,
                                        final INDArray label) {
    final int lastLayerIndex = layers.length - 1;
    final INDArray gradient = layers[lastLayerIndex].backPropagate(activations.get(lastLayerIndex + 1), label);
    final List<INDArray> gradients =
        backPropagateFromTo(lastLayerIndex - 1, to, activations, derivatives, gradient);
    gradients.add(gradient);
    return gradients;
  }

  /**
   * Computes gradients from the specified beginning layer and the specified ending layer.
   * @param begin the index of beginning layer.
   * @param end the index of ending layer.
   * @param activations a list of activations of each layer.
   * @param derivatives a list of derivatives of each layer.
   * @param nextGradient the gradient vector for next layer.
   * @return a list of gradients for each layer.
   */
  public List<INDArray> backPropagateFromTo(final int begin, final int end,
                                            final List<INDArray> activations,
                                            final List<INDArray> derivatives,
                                            final INDArray nextGradient) {
    if (begin == layers.length - 1) {
      throw new RuntimeException("The beginning layer of backPropagateFromTo cannot be output layer");
    }

    final List<INDArray> gradients = new ArrayList<>();
    INDArray gradient = nextGradient;

    for (int i = begin; i >= end; --i) {
      gradient = layers[i].backPropagate(
          activations.get(i + 1), derivatives.get(i + 1), layers[i + 1].getLayerParameter(), gradient);
      gradients.add(gradient);
    }

    Collections.reverse(gradients);
    return gradients;
  }
}
