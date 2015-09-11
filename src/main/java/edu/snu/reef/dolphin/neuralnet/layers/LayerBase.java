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
 * Abstract class for the layer of a neural network.
 */
public abstract class LayerBase {

  private final int index;
  private final int numOutput;
  private LayerParameter layerParameter;

  protected LayerBase(final int index, final int numOutput) {
    this.index = index;
    this.numOutput = numOutput;
  }

  /**
   * @return the index of the layer.
   */
  public final int getIndex() {
    return this.index;
  }

  /**
   * @return the number of layer output nodes.
   */
  public final int getNumOutput() {
    return this.numOutput;
  }

  /**
   * Replaces the parameter of the layer.
   * @param layerParameter a new parameter of the layer.
   */
  public final void setLayerParameter(final LayerParameter layerParameter) {
    if (!isLearnable()) {
      throw new RuntimeException(this + " is not a learnable layer. setLayerParameter() should not be called.");
    }

    this.layerParameter = layerParameter;
  }

  /**
   * @return the parameter of the layer.
   */
  public final LayerParameter getLayerParameter() {
    if (!isLearnable()) {
      throw new RuntimeException(this + " is not a learnable layer. getLayerParameter() should not be called.");
    }

    return this.layerParameter;
  }

  /**
   * @return whether this layer can learn from training data or not.
   */
  public abstract boolean isLearnable();

  /**
   * Applies the derivative of the activation function of the layer to each element of matrix.
   * @param activation the activations of the layer.
   * @return the derivatives for the given activation.
   */
  public abstract INDArray derivative(final INDArray activation);

  /**
   * Computes the activations.
   * @param input the input vector for the layer.
   * @return the activations.
   */
  public abstract INDArray feedForward(final INDArray input);

  /**
   * Computes the error.
   * @param activation the activation values.
   * @param nextParameter the parameter of the next layer - the one closer to the output layer.
   * @param nextError the error of the next layer - the one closer to the output layer.
   * @return the error for this layer with the specified activation values.
   */
  public abstract INDArray backPropagate(final INDArray activation,
                                         final LayerParameter nextParameter,
                                         final INDArray nextError);

  /**
   * Computes the error. (only for output layer)
   * @param activation the activations for output layer.
   * @param label the expected output.
   * @return the error for the output layer with the specified label.
   */
  public abstract INDArray backPropagate(final INDArray activation,
                                         final INDArray label);

}
