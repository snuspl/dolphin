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

import edu.snu.dolphin.dnn.conf.ActivationLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.util.Nd4jUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

/**
 * Test class for activation layer.
 */
public final class ActivationLayerTest {

  private static final float TOLERANCE = 1e-6f;

  private final INDArray input = Nd4j.create(new float[][]{{-1.0f, -0.5f, 0.5f, 1.0f}, {-0.6f, -0.3f, 0.3f, 0.6f}});
  private final INDArray expectedSigmoidActivation = Nd4j.create(new float[][]{
      {2.689414214e-01f, 3.775406688e-01f, 6.224593312e-01f, 7.310585786e-01f},
      {3.543436938e-01f, 4.255574832e-01f, 5.744425168e-01f, 6.456563062e-01f}});
  private final INDArray nextError = Nd4j.create(new float[][]{
      {0.1f, 0.5f, -0.2f, 0.3f},
      {0.18f, -0.23f, 0.195f, -0.076f}});
  private final INDArray expectedSigmoidError = Nd4j.create(new float[][]{
      {1.96611933241e-02f, 1.17501856101e-01f, -4.70007424403e-02f, 5.89835799724e-02f},
      {4.11811632822e-02f, -5.62254116889e-02f, 4.76693707797e-02f, -1.73876022747e-02f}});

  private LayerBase sigmoidActivationLayer;

  public ActivationLayerTest() {
  }

  @Before
  public void setup() throws InjectionException {
    final Configuration layerConf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.LayerIndex.class, String.valueOf(0))
        .build();

    final ActivationLayerConfigurationBuilder builder = ActivationLayerConfigurationBuilder.newConfigurationBuilder()
        .setNumInput(input.length())
        .setNumOutput(expectedSigmoidActivation.length());

    this.sigmoidActivationLayer =
        Tang.Factory.getTang().newInjector(layerConf, builder.setActivationFunction("sigmoid").build())
        .getInstance(LayerBase.class);
  }

  @Test
  public void testSigmoidActivation() {
    final INDArray activation = sigmoidActivationLayer.feedForward(input);
    assertTrue(Nd4jUtils.equals(expectedSigmoidActivation, activation, TOLERANCE));
  }

  @Test
  public void testSigmoidBackPropagate() {
    final INDArray error = sigmoidActivationLayer.backPropagate(input, expectedSigmoidActivation, nextError);
    assertTrue(Nd4jUtils.equals(expectedSigmoidError, error, TOLERANCE));
  }
}
