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
package edu.snu.dolphin.dnn.layerparam.initializer;

import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

import static edu.snu.dolphin.dnn.util.NeuralNetworkUtils.shapeFromString;

/**
 * Default parameter initializer.
 *
 * This initializer is for layers which do not have layer parameters (i.e. not learnable layer).
 * This initializer is used for layers whose output shape is equal to input shape.
 */
public final class DefaultLayerParameterInitializer implements LayerParameterInitializer {

  private final int index;
  private final int[] inputShape;
  private final LayerParameter emptyLayerParam;

  @Inject
  public DefaultLayerParameterInitializer(
      final MatrixFactory matrixFactory,
      @Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
      @Parameter(LayerConfigurationParameters.LayerInputShape.class) final String inputShape) {
    this.index = index;
    this.inputShape = shapeFromString(inputShape);
    this.emptyLayerParam = LayerParameter.newEmptyInstance(matrixFactory);
  }

  /**
   * @return the initial parameter of the layer.
   */
  public LayerParameter generateInitialParameter() {
    return emptyLayerParam;
  }

  /**
   * @return the index of the layer.
   */
  public int getIndex() {
    return index;
  }

  @Override
  public int[] getOutputShape() {
    return inputShape;
  }

}
