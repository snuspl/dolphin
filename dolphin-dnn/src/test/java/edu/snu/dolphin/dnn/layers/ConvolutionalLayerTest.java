/*
 * Copyright (C) 2016 Seoul National University
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
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.conf.*;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.*;

import static edu.snu.dolphin.dnn.layers.LayerParameterUtils.compare;
import static org.junit.Assert.assertTrue;

/**
 * Test class for convolutional layer.
 */
public class ConvolutionalLayerTest {

  private static MatrixFactory matrixFactory;
  private static final float TOLERANCE = 1e-4f;

  static {
    final Configuration configuration = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();
    try {
      matrixFactory = Tang.Factory.getTang().newInjector(configuration).getInstance(MatrixFactory.class);
    } catch (final InjectionException e) {
      throw new RuntimeException("InjectionException while injecting a matrix factory: " + e);
    }
  }

  private final Matrix input = matrixFactory.create(new float[][]{
      {0, 10},
      {9, 22},
      {2, 13},
      {8, 0},
      {3, 5},
      {7, 3},
      {10, 6},
      {4, 7},
      {1, 9}});
  private final Matrix expectedConvolutionalActivation = matrixFactory.create(new float[][]{
      {0.406522f, -2.8708f},
      {-0.059460f, -4.5747f},
      {-0.238702f, 1.6502f},
      {-0.760506f, 1.4035f}});
  private final Matrix expectedConvolutionalWithPaddingActivation = matrixFactory.create(new float[][]{
      {0.00000f, 2.4788705f},
      {2.23098f, 6.1111124f},
      {1.08761f, 4.6692458f},
      {0.13152f, 0.8548766f},
      {1.98310f, -0.9591301f},
      {0.40652f, -2.8707870f},
      {-0.05946f, -4.5747085f},
      {0.06029f, -2.4028976f},
      {1.71157f, 1.4873223f},
      {-0.23870f, 1.6502027f},
      {-0.76051f, 1.4034946f},
      {-1.33434f, -0.0082032f},
      {-0.95913f, -0.5754781f},
      {-2.38379f, -1.8714727f},
      {-0.89597f, -2.2633123f},
      {-0.20001f, -1.8001224f}});
  private final Matrix nextError = matrixFactory.create(new float[][]{
      {0.1f, 0},
      {0.3f, 0.3f},
      {0.6f, 0.4f},
      {0.4f, 0.1f}});
  private final Matrix nextErrorWithPadding = matrixFactory.create(new float[][]{
      {0.2f, 0},
      {0.1f, 0.4f},
      {0.4f, 1.2f},
      {0.8f, 0.1f},
      {0.2f, 0},
      {0, 0.3f},
      {0.4f, 1.3f},
      {0.7f, 0.8f},
      {0.8f, 0.6f},
      {0.1f, 0.5f},
      {0, 0.1f},
      {0.2f, 0.2f},
      {0.2f, 0.5f},
      {0.1f, 0},
      {0.7f, 0.2f},
      {0.5f, 0.3f}});
  private final Matrix expectedConvolutionalError = matrixFactory.create(new float[][]{
      {-0.0200014f, 0},
      {-0.0695954f, -0.0600042f},
      {-0.0287739f, -0.0287739f},
      {-0.1134325f, -0.0800056f},
      {-0.0930369f, -0.0386389f},
      {0.03600091f, 0.06477481f},
      {0.0394554f, 0.0263036f},
      {0.1750358f, 0.1057307f},
      {0.0991548f, 0.0247887f}});
  private final Matrix expectedConvolutionalWithPaddingError = matrixFactory.create(new float[][]{
      {0.0369707f, -0.0337006f},
      {-0.0289132f, -0.110726f},
      {-0.0266129f, 0.0193425f},
      {-0.0471544f, -0.137827f},
      {0.0167122f, 0.0918946f},
      {0.105183f, 0.325266f},
      {0.165702f, 0.133655f},
      {-0.124812f, 0.0905167f},
      {-0.153994f, -0.0412462f}});

  private final LayerParameter expectedConvolutionalLayerParams =
      LayerParameter.newBuilder()
      .setWeightParam(matrixFactory.create(new float[]
          {15.8f, 12.29999f, 13.9f, 9.8f}).transpose())
      .setBiasParam(matrixFactory.create(new float[]
          {0.1f, 0.6f, 1, 0.5f}).transpose())
      .build();

  private final LayerParameter expectedConvolutionalLayerWithPaddingParams =
      LayerParameter.newBuilder()
      .setWeightParam(matrixFactory.create(new float[]
          {58.7f, 41.699997f, 58.6f, 52.300003f}).transpose())
      .setBiasParam(matrixFactory.create(new float[]
          {0.2f, 0.5f, 1.6f, 0.9f, 0.2f, 0.3f, 1.7f, 1.5f, 1.4f, 0.6f, 0.1f, 0.4f, 0.7f, 0.1f, 0.9f, 0.8f}).transpose())
      .build();

  private LayerBase convolutionalLayer;
  private LayerBase convolutionalWithPaddingLayer;

  @Before
  public void setup() throws InjectionException {
    final Configuration layerConf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerIndex.class, String.valueOf(0))
        .bindNamedParameter(LayerInputShape.class, "3,3")
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();

    final ConvolutionalLayerConfigurationBuilder builder = ConvolutionalLayerConfigurationBuilder
        .newConfigurationBuilder()
        .setInitWeight(0.2f)
        .setInitBias(0)
        .setRandomSeed(10)
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    final ConvolutionalLayerConfigurationBuilder builderWithPadding = ConvolutionalLayerConfigurationBuilder
        .newConfigurationBuilder()
        .setInitWeight(0.2f)
        .setInitBias(0)
        .setRandomSeed(10)
        .setPaddingHeight(1)
        .setPaddingWidth(1)
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    this.convolutionalLayer =
        Tang.Factory.getTang().newInjector(layerConf, builder.build())
        .getInstance(LayerBase.class);

    this.convolutionalWithPaddingLayer =
        Tang.Factory.getTang().newInjector(layerConf, builderWithPadding.build())
        .getInstance(LayerBase.class);
  }

  @Test
  public void testConvolutionalActivation() {
    final Matrix convolutionalActivation = convolutionalLayer.feedForward(input);
    assertTrue(expectedConvolutionalActivation.compare(convolutionalActivation, TOLERANCE));
  }

  @Test
  public void testConvolutionalBackPropagate() {
    final Matrix error = convolutionalLayer.backPropagate(input, expectedConvolutionalActivation, nextError);
    assertTrue(expectedConvolutionalError.compare(error, TOLERANCE));
  }

  @Test
  public void testConvolutionalWithPaddingActivation() {
    final Matrix convolutionalActivation = convolutionalWithPaddingLayer.feedForward(input);
    assertTrue(expectedConvolutionalWithPaddingActivation.compare(convolutionalActivation, TOLERANCE));
  }

  @Test
  public void testConvolutionalWithPaddingBackPropagate() {
    final Matrix error = convolutionalWithPaddingLayer.backPropagate(
        input, expectedConvolutionalWithPaddingActivation, nextErrorWithPadding);
    assertTrue(expectedConvolutionalWithPaddingError.compare(error, TOLERANCE));
  }

  @Test
  public void testConvolutionalGradient() {
    final LayerParameter convolutionalLayerParams = convolutionalLayer.generateParameterGradient(input, nextError);
    assertTrue(compare(expectedConvolutionalLayerParams, convolutionalLayerParams, TOLERANCE));
  }

  @Test
  public void testConvolutionalWithPaddingGradient() {
    final LayerParameter convolutionalLayerParams =
        convolutionalWithPaddingLayer.generateParameterGradient(input, nextErrorWithPadding);
    assertTrue(compare(expectedConvolutionalLayerWithPaddingParams, convolutionalLayerParams, TOLERANCE));
  }
}
