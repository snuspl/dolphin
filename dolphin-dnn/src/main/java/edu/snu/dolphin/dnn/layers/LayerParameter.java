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

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;

/**
 * The parameter of the layer.
 */
public final class LayerParameter {
  private final Matrix weightParam;
  private final Matrix biasParam;
  private final String poolingType;
  private final int stride;
  private final int kernalHeight;
  private final int kernalWidth;

  /**
   * Generates a new instance of a layer parameter.
   * @param matrixFactory a factory to create the matrices that the layer parameter contains
   * @return the generated empty layer parameter
   */
  public static LayerParameter newEmptyInstance(final MatrixFactory matrixFactory) {
    return new LayerParameter(matrixFactory.create(0), matrixFactory.create(0), null, 0, 0, 0);
  }

  /**
   * @return a new LayerParameter builder.
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * @return the weight matrix of the parameter.
   */
  public Matrix getWeightParam() {
    return weightParam;
  }

  /**
   * @return the bias vector of the parameter.
   */
  public Matrix getBiasParam() {
    return biasParam;
  }

  public String getPoolingType() {
    return poolingType;
  }

  public int getStride() {
    return stride;
  }

  public int getKernalHeight() {
    return kernalHeight;
  }

  public int getKernalWidth() {
    return kernalWidth;
  }

  public static final class Builder implements org.apache.reef.util.Builder<LayerParameter> {
    private Matrix weightParam;
    private Matrix biasParam;
    private String poolingType;
    private int stride;
    private int kernalHeight;
    private int kernalWidth;

    public Builder setWeightParam(final Matrix weightParam) {
      this.weightParam = weightParam;
      return this;
    }

    public Builder setBiasParam(final Matrix biasParam) {
      this.biasParam = biasParam;
      return this;
    }

    public Builder setPoolingType(final String poolingType) {
      this.poolingType = poolingType;
      return this;
    }

    public Builder setStride(final int stride) {
      this.stride = stride;
      return this;
    }

    public Builder setKernalHeight(final int kernalHeight) {
      this.kernalHeight = kernalHeight;
      return this;
    }

    public Builder setKernalWidth(final int kernalWidth) {
      this.kernalWidth = kernalWidth;
      return this;
    }

    @Override
    public LayerParameter build() {
      return new LayerParameter(this.weightParam, this.biasParam,
          this.poolingType, this.stride, this.kernalHeight, this.kernalWidth);
    }
  }

  private LayerParameter(final Matrix weightParam,
                         final Matrix biasParam,
                         final String poolingType,
                         final int stride,
                         final int kernalHeight,
                         final int kernalWidth) {
    this.weightParam = weightParam;
    this.biasParam = biasParam;
    this.poolingType = poolingType;
    this.stride = stride;
    this.kernalHeight = kernalHeight;
    this.kernalWidth = kernalWidth;
  }

  @Override
  public String toString() {
    return "weight: " + weightParam.toString() + ", bias: " + biasParam.toString()
        + ", pooling type: " + poolingType + ", stride: " + Integer.toString(stride)
        + ", kernal height: " + Integer.toString(kernalHeight) + ", kernal width: " + Integer.toString(kernalWidth);
  }

  @Override
  public boolean equals(final Object obj) {
    if (!(obj instanceof LayerParameter)) {
      return false;
    }

    final LayerParameter other = (LayerParameter)obj;
    return weightParam.equals(other.weightParam) && biasParam.equals(other.biasParam)
        && poolingType.equals(other.poolingType) && this.stride == other.stride
        && this.kernalHeight == other.kernalHeight && this.kernalWidth == other.kernalWidth;
  }

  @Override
  public int hashCode() {
    final int weightParamHashCode = weightParam == null ? 0 : weightParam.hashCode();
    final int biasParamHashCode = biasParam == null ? 0 : biasParam.hashCode();
    final int poolingTypeHashCode = poolingType == null ? 0 : poolingType.hashCode();
    final int strideHashCode = stride;
    final int kernalHeightHashCode = kernalHeight;
    final int kernalWidthHashCode = kernalWidth;
    return strideHashCode * (int) Math.pow(31, 5) + kernalHeightHashCode * (int) Math.pow(31, 4) +
        kernalWidthHashCode * (int) Math.pow(31, 3) + poolingTypeHashCode * (int) Math.pow(31, 2) +
        weightParamHashCode * 31 + biasParamHashCode;
  }
}
