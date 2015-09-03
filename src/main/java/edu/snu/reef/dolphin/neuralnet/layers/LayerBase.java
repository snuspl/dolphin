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
 * A abstract base class having common functions for all layers.
 */
public abstract class LayerBase implements Layer {

  /**
   * The index of layer.
   */
  protected final int index;

  /**
   * The number of output.
   */
  private final int numOutput;

  /**
   * The parameter (weights, bias) of layer.
   */
  private LayerParameter layerParameter;

  /**
   * A constructor with common parameters.
   *
   * @param index
   * @param numOutput
   */
  public LayerBase(final int index, final int numOutput) {
    this.index = index;
    this.numOutput = numOutput;
  }

  /**
   * {@inheritDoc}
   */
  public int getIndex() {
    return index;
  }

  /**
   * {@inheritDoc}
   */
  public int getNumOutput() {
    return numOutput;
  }

  /**
   * {@inheritDoc}
   */
  public void setLayerParameter(final LayerParameter layerParameter) {
    this.layerParameter = layerParameter;
  }

  /**
   * {@inheritDoc}
   */
  public LayerParameter getLayerParameter() {
    return this.layerParameter;
  }

  /**
   * {@inheritDoc}
   */
  public abstract INDArray derivative(final INDArray activation);

  /**
   * {@inheritDoc}
   */
  public abstract INDArray feedForward(final INDArray input);

  /**
   * {@inheritDoc}
   */
  public abstract INDArray backPropagate(final INDArray activation,
                                         final INDArray derivative,
                                         final LayerParameter prevParam,
                                         final INDArray nextGradient);

  /**
   * {@inheritDoc}
   */
  public abstract INDArray backPropagate(final INDArray activation, final INDArray label);

  /**
   * {@inheritDoc}
   */
  public boolean isLearnable() {
    return true;
  }
}
