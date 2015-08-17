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
 * The parameter of the layer.
 */
public final class LayerParameter {
  private final INDArray weightParam;
  private final INDArray biasParam;

  /**
   * Returns the new LayerParameter builder.
   * @return
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * Returns the weight matrix of the parameter.
   * @return
   */
  public INDArray getWeightParam() {
    return weightParam;
  }

  /**
   * Returns the bias vector of the parameter.
   * @return
   */
  public INDArray getBiasParam() {
    return biasParam;
  }

  public static final class Builder implements org.apache.reef.util.Builder<LayerParameter> {
    private INDArray weightParam;
    private INDArray biasParam;

    public Builder setWeightParam(final INDArray weightParam) {
      this.weightParam = weightParam;
      return this;
    }

    public Builder setBiasParam(final INDArray biasParam) {
      this.biasParam = biasParam;
      return this;
    }

    @Override
    public LayerParameter build() {
      return new LayerParameter(this.weightParam, this.biasParam);
    }
  }

  LayerParameter(final INDArray weightParam,
                 final INDArray biasParam) {
    this.weightParam = weightParam;
    this.biasParam = biasParam;
  }

  @Override
  public String toString() {
    return "weight: " + weightParam.toString() + ", bias: " + biasParam.toString();
  }
}
