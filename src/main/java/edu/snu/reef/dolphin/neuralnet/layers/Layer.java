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

import org.jblas.DoubleMatrix;

/**
 * Interface for the layer of a neural network.
 */
public interface Layer {

  /**
   * Returns the index of the layer.
   * @return
   */
  int getIndex();

  /**
   * Replaces the parameter of the layer.
   * @param layerParameter
   */
  void setLayerParameter(final LayerParameter layerParameter);

  /**
   * Returns the parameter of the layer.
   * @return
   */
  LayerParameter getLayerParameter();

  /**
   * Applies the derivative of the activation function of the layer to each element of matrix.
   * @param activation
   * @return
   */
  DoubleMatrix derivative(final DoubleMatrix activation);

  /**
   * Computes the activation values.
   * @param input
   * @return
   */
  DoubleMatrix feedForward(final DoubleMatrix input);

  /**
   * Computes the gradients.
   * @param activation the activation values.
   * @param derivative the derivatives of activation function.
   * @param prevParam the parameter of the previous layer.
   * @param nextGradient the gradients of the next layer.
   * @return
   */
  DoubleMatrix backPropagate(final DoubleMatrix activation,
                             final DoubleMatrix derivative,
                             final LayerParameter prevParam,
                             final DoubleMatrix nextGradient);

  /**
   * Computes the gradients. (only for output layer)
   * @param activation
   * @param label
   * @return
   */
  DoubleMatrix backPropagate(final DoubleMatrix activation,
                             final DoubleMatrix label);
}