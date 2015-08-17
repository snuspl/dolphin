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
 * The parameter of the layer.
 */
public final class LayerParameter {
  private final DoubleMatrix weightParam;
  private final DoubleMatrix biasParam;

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
  public DoubleMatrix getWeightParam() {
    return weightParam;
  }

  /**
   * Returns the bias vector of the parameter.
   * @return
   */
  public DoubleMatrix getBiasParam() {
    return biasParam;
  }

  public static final class Builder implements org.apache.reef.util.Builder<LayerParameter> {
    private DoubleMatrix weightParam;
    private DoubleMatrix biasParam;

    public Builder setWeightParam(final DoubleMatrix weightParam) {
      this.weightParam = weightParam;
      return this;
    }

    public Builder setBiasParam(final DoubleMatrix biasParam) {
      this.biasParam = biasParam;
      return this;
    }

    @Override
    public LayerParameter build() {
      return new LayerParameter(this.weightParam, this.biasParam);
    }
  }

  LayerParameter(final DoubleMatrix weightParam,
                 final DoubleMatrix biasParam) {
    this.weightParam = weightParam;
    this.biasParam = biasParam;
  }

  @Override
  public String toString() {
    return "weight: " + weightParam.toString() + ", bias: " + biasParam.toString();
  }
}
