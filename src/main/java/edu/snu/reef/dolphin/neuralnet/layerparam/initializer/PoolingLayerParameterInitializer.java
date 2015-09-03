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
package edu.snu.reef.dolphin.neuralnet.layerparam.initializer;

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Dummy parameter Initializer of pooling layer.
 * <p/>
 * pooling layer has not weights and bias.
 */
public final class PoolingLayerParameterInitializer implements LayerParameterInitializer {

  private final int index;

  @Inject
  public PoolingLayerParameterInitializer(@Parameter(LayerConfigurationParameters.LayerIndex.class) final int index) {
    this.index = index;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public LayerParameter generateInitialParameter() {
    return LayerParameter.newBuilder().build();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int getIndex() {
    return this.index;
  }
}
