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
package edu.snu.dolphin.dnn.layers;

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
   * Applies a derivative of the activation function of the layer to each element of matrix.
   * @param activation an activation value of the layer.
   * @return a derivative for the given activation.
   */
  public abstract INDArray derivative(final INDArray activation);

  /**
   * Computes an activation value.
   * @param input an input value for the layer.
   * @return a activation value.
   */
  public abstract INDArray feedForward(final INDArray input);

  /**
   * Computes an error.
   * @param activation an activation value.
   * @param nextParameter the parameter of the next layer - the one closer to the output layer.
   * @param nextError an error of the next layer - the one closer to the output layer.
   * @return an error for this layer with the specified activation value.
   */
  public abstract INDArray backPropagate(final INDArray activation,
                                         final LayerParameter nextParameter,
                                         final INDArray nextError);

  /**
   * Computes the error. (only for output layer)
   * @param activation an activation for output layer.
   * @param label the expected output.
   * @return an error for the output layer with the specified label.
   */
  public abstract INDArray backPropagate(final INDArray activation,
                                         final INDArray label);

  /**
   * Computes a parameter gradient for this layer.
   * @param input an input value for this layer.
   * @param error an error for this layer.
   * @return a parameter gradient for this layer.
   */
  public abstract LayerParameter generateParameterGradient(final INDArray input, final INDArray error);
}
