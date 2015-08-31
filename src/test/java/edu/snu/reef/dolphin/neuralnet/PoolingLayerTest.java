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
package edu.snu.reef.dolphin.neuralnet;

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.conf.PoolingLayerConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.layers.PoolingLayer;
import edu.snu.reef.dolphin.neuralnet.util.Nd4jUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

/**
 * Unit tests for PoolingLayer.
 */
public class PoolingLayerTest {

  private final float tolerance = 1e-6f;

  private final INDArray input = Nd4j.create(new float[]{
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
  }, new int[]{4, 4}).transposei();

  private final INDArray maxOutput = Nd4j.create(new float[]{
      6, 8,
      14, 16
  }, new int[]{2, 2}).transposei();

  private final INDArray maxDerivative = Nd4j.create(new float[]{
      0, 0, 0, 0,
      0, 1, 0, 1,
      0, 0, 0, 0,
      0, 1, 0, 1
  }, new int[]{4, 4}).transposei();

  private final INDArray meanOutput = Nd4j.create(new float[]{
      3.5f, 5.5f,
      11.5f, 13.5f
  }, new int[]{2, 2}).transposei();

  private final INDArray meanDerivative = Nd4j.create(new float[]{
      0.25f, 0.25f, 0.25f, 0.25f,
      0.25f, 0.25f, 0.25f, 0.25f,
      0.25f, 0.25f, 0.25f, 0.25f,
      0.25f, 0.25f, 0.25f, 0.25f
  }, new int[]{4, 4}).transposei();

  private final Configuration maxPoolingLayerConf = PoolingLayerConfigurationBuilder.newConfigurationBuilder()
      .setNumInput(input.length())
      .setNumOutput(maxOutput.length())
      .setInitWeight(0.0001f)
      .setInitBias(0.0002f)
      .setRandomSeed(10)
      .setPoolingSize(2)
      .setPoolingFunction("max")
      .setActivationFunction("none")
      .build();

  private final Configuration meanPoolingLayerConf = PoolingLayerConfigurationBuilder.newConfigurationBuilder()
      .setNumInput(input.length())
      .setNumOutput(meanOutput.length())
      .setInitWeight(0.0001f)
      .setInitBias(0.0002f)
      .setRandomSeed(10)
      .setPoolingSize(2)
      .setPoolingFunction("mean")
      .setActivationFunction("none")
      .build();

  private final Configuration maxConf = Configurations.merge(maxPoolingLayerConf,
      Tang.Factory.getTang().newConfigurationBuilder()
          .bindNamedParameter(LayerConfigurationParameters.LayerIndex.class, String.valueOf(0))
          .build());

  private final Configuration meanConf = Configurations.merge(meanPoolingLayerConf,
      Tang.Factory.getTang().newConfigurationBuilder()
          .bindNamedParameter(LayerConfigurationParameters.LayerIndex.class, String.valueOf(0))
          .build());

  private PoolingLayer maxPoolingLayer;

  private PoolingLayer meanPoolingLayer;

  @Before
  public void buildLayer() throws InjectionException {
    final Injector injector1 = Tang.Factory.getTang().newInjector(maxConf);
    maxPoolingLayer = injector1.getInstance(PoolingLayer.class);
    final Injector injector2 = Tang.Factory.getTang().newInjector(meanConf);
    meanPoolingLayer = injector2.getInstance(PoolingLayer.class);
  }

  /**
   * Unit test for feed forward of max pooling layer.
   */
  @Test
  public void maxPoolingFeedForwardTest() {
    final INDArray actual = maxPoolingLayer.feedForward(input);
    assertTrue(Nd4jUtils.equals(actual, maxOutput, tolerance));
  }

  /**
   * Unit test for derivative of max pooling layer.
   */
  @Test
  public void maxPoolingDerivativeTest() {
    final INDArray activation = maxPoolingLayer.feedForward(input);
    final INDArray derivative = maxPoolingLayer.derivative(activation);
    assertTrue(Nd4jUtils.equals(derivative, maxDerivative, tolerance));
  }

  /**
   * Unit test for feed forward of mean pooling layer.
   */
  @Test
  public void meanPoolingFeedForwardTest() {
    final INDArray actual = meanPoolingLayer.feedForward(input);
    assertTrue(Nd4jUtils.equals(actual, meanOutput, tolerance));
  }

  /**
   * Unit test for derivative of mean pooling layer.
   */
  @Test
  public void meanPoolingDerivativeTest() {
    final INDArray activation = meanPoolingLayer.feedForward(input);
    final INDArray derivative = meanPoolingLayer.derivative(activation);
    assertTrue(Nd4jUtils.equals(derivative, meanDerivative, tolerance));
  }
}
