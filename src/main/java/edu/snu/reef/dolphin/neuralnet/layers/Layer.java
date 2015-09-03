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
package edu.snu.reef.dolphin.neuralnet.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for the layer of a neural network.
 */
public interface Layer {

  /**
   * @return the index of the layer.
   */
  int getIndex();

  /**
   * @return the number of layer output nodes.
   */
  int getNumOutput();

  /**
   * Replaces the parameter of the layer.
   * @param layerParameter a new parameter of the layer.
   */
  void setLayerParameter(final LayerParameter layerParameter);

  /**
   * @return the parameter of the layer.
   */
  LayerParameter getLayerParameter();

  /**
   * Applies the derivative of the activation function of the layer to each element of matrix.
   * @param activation the activations of the layer.
   * @return the derivatives for the given activation.
   */
  INDArray derivative(final INDArray activation);

  /**
   * Computes the activations.
   * @param input the input vector for the layer.
   * @return the activations.
   */
  INDArray feedForward(final INDArray input);

  /**
   * Computes the gradients.
   * @param activation the activation values.
   * @param derivative the derivatives of activation function.
   * @param prevParam the parameter of the previous layer.
   * @param nextGradient the gradients of the next layer.
   * @return the gradients for the specified activations and derivatives.
   */
  INDArray backPropagate(final INDArray activation,
                         final INDArray derivative,
                         final LayerParameter prevParam,
                         final INDArray nextGradient);

  /**
   * Computes the gradients. (only for output layer)
   * @param activation the activations for output layer.
   * @param label the expected output.
   * @return the gradients for output layer.
   */
  INDArray backPropagate(final INDArray activation,
                         final INDArray label);

  /**
   * @return true if the layer is learnable.
   */
  boolean isLearnable();
}
