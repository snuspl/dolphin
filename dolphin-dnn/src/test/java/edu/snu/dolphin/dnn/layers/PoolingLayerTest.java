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
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.conf.*;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Test;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters.*;
import org.junit.runners.MethodSorters;

import static org.junit.Assert.assertTrue;

/**
 * Test class for pooling layer.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public final class PoolingLayerTest {

  private static MatrixFactory matrixFactory;
  private static final float TOLERANCE = 1e-6f;

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
  private final Matrix expectedMaxPoolingActivation = matrixFactory.create(new float[][]{
      {9, 22},
      {9, 22},
      {10, 7},
      {7, 9}});
  private final Matrix expectedAveragePoolingActivation = matrixFactory.create(new float[][]{
      {5, 9.25f},
      {5.25f, 10.75f},
      {6.25f, 4.5f},
      {3.75f, 6}});
  private final Matrix expectedRemainderExistingMaxPoolingActivation = matrixFactory.create(new float[][]{
      {9, 22},
      {7, 13},
      {10, 7},
      {1, 9}});
  private final Matrix expectedRemainderExistingAveragePoolingActivation = matrixFactory.create(new float[][]{
      {5, 9.25f},
      {4.5f, 8},
      {7, 6.5f},
      {1, 9}});
  private final Matrix expectedMaxPoolingWithPaddingActivation = matrixFactory.create(new float[][]{
      {0, 10},
      {9, 22},
      {9, 22},
      {2, 13},
      {8, 10},
      {9, 22},
      {9, 22},
      {7, 13},
      {10, 6},
      {10, 7},
      {7, 9},
      {7, 9},
      {10, 6},
      {10, 7},
      {4, 9},
      {1, 9}});
  private final Matrix expectedAveragePoolingWithPaddingActivation = matrixFactory.create(new float[][]{
      {0, 2.5f},
      {2.25f, 8},
      {2.75f, 8.75f},
      {0.5f, 3.25f},
      {2, 2.5f},
      {5, 9.25f},
      {5.25f, 10.75f},
      {2.25f, 4},
      {4.5f, 1.5f},
      {6.25f, 4.5f},
      {3.75f, 6},
      {2, 3},
      {2.5f, 1.5f},
      {3.5f, 3.25f},
      {1.25f, 4},
      {0.25f, 2.25f}});
  private final Matrix nextError = matrixFactory.create(new float[][]{
      {12, 0},
      {16, 4},
      {20, 12},
      {4, 8}});
  private final Matrix nextErrorWithPadding = matrixFactory.create(new float[][]{
      {12, 0},
      {16, 4},
      {20, 12},
      {4, 8},
      {0, 8},
      {8, 20},
      {4, 12},
      {4, 0},
      {8, 4},
      {4, 0},
      {0, 16},
      {12, 0},
      {16, 4},
      {12, 16},
      {0, 4},
      {4, 8}});
  private final Matrix expectedMaxPoolingError = matrixFactory.create(new float[][]{
      {0, 0},
      {28, 4},
      {0, 0},
      {0, 0},
      {0, 0},
      {4, 0},
      {20, 0},
      {0, 12},
      {0, 8}});
  private final Matrix expectedAveragePoolingError = matrixFactory.create(new float[][]{
      {3, 0},
      {7, 1},
      {4, 1},
      {8, 3},
      {13, 6},
      {5, 3},
      {5, 3},
      {6, 5},
      {1, 2}});
  private final Matrix expectedRemainderExistingMaxPoolingError = matrixFactory.create(new float[][]{
      {0, 0},
      {12, 0},
      {0, 4},
      {0, 0},
      {0, 0},
      {16, 0},
      {20, 0},
      {0, 12},
      {4, 8}});
  private final Matrix expectedRemainderExistingAveragePoolingError = matrixFactory.create(new float[][]{
      {3, 0},
      {3, 0},
      {8, 2},
      {3, 0},
      {3, 0},
      {8, 2},
      {10, 6},
      {10, 6},
      {4, 8}});
  private final Matrix expectedMaxPoolingWithPaddingError = matrixFactory.create(new float[][]{
      {12, 8},
      {48, 48},
      {4, 8},
      {0, 0},
      {0, 0},
      {16, 0},
      {40, 8},
      {0, 16},
      {4, 28}});
  private final Matrix expectedAveragePoolingWithPaddingError = matrixFactory.create(new float[][]{
      {9, 8},
      {12, 12},
      {8, 8},
      {5, 8},
      {4, 12},
      {5, 7},
      {10, 6},
      {4, 9},
      {4, 7}});

  private LayerBase maxPoolingLayer;
  private LayerBase averagePoolingLayer;
  private LayerBase remainderExistingMaxPoolingLayer;
  private LayerBase remainderExistingAveragePoolingLayer;
  private LayerBase maxPoolingWithPaddingLayer;
  private LayerBase averagePoolingWithPaddingLayer;

  @Before
  public void setup() throws InjectionException {
    final Configuration layerConf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerIndex.class, String.valueOf(0))
        .bindNamedParameter(LayerInputShape.class, "3,3")
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();

    final PoolingLayerConfigurationBuilder maxBuilder = PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("MAX")
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    final PoolingLayerConfigurationBuilder averageBuilder = PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("AVERAGE")
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    final PoolingLayerConfigurationBuilder remainderExistingMaxBuilder =
        PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("MAX")
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(2)
        .setStrideWidth(2);

    final PoolingLayerConfigurationBuilder remainderExistingAverageBuilder =
        PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("AVERAGE")
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(2)
        .setStrideWidth(2);

    final PoolingLayerConfigurationBuilder maxWithPaddingBuilder =
        PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("MAX")
        .setPaddingHeight(1)
        .setPaddingWidth(1)
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    final PoolingLayerConfigurationBuilder averageWithPaddingBuilder =
        PoolingLayerConfigurationBuilder.newConfigurationBuilder()
        .setPoolingType("AVERAGE")
        .setPaddingHeight(1)
        .setPaddingWidth(1)
        .setKernelHeight(2)
        .setKernelWidth(2)
        .setStrideHeight(1)
        .setStrideWidth(1);

    this.maxPoolingLayer =
        Tang.Factory.getTang().newInjector(layerConf, maxBuilder.build())
        .getInstance(LayerBase.class);

    this.averagePoolingLayer =
        Tang.Factory.getTang().newInjector(layerConf, averageBuilder.build())
        .getInstance(LayerBase.class);

    this.remainderExistingMaxPoolingLayer =
        Tang.Factory.getTang().newInjector(layerConf, remainderExistingMaxBuilder.build())
        .getInstance(LayerBase.class);

    this.remainderExistingAveragePoolingLayer =
        Tang.Factory.getTang().newInjector(layerConf, remainderExistingAverageBuilder.build())
        .getInstance(LayerBase.class);

    this.maxPoolingWithPaddingLayer =
        Tang.Factory.getTang().newInjector(layerConf, maxWithPaddingBuilder.build())
        .getInstance(LayerBase.class);

    this.averagePoolingWithPaddingLayer =
        Tang.Factory.getTang().newInjector(layerConf, averageWithPaddingBuilder.build())
        .getInstance(LayerBase.class);
  }

  @Test
  public void testMaxPoolingActivation() {
    final Matrix poolingActivation = maxPoolingLayer.feedForward(input);
    assertTrue(expectedMaxPoolingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testMaxPoolingBackPropagate() {
    maxPoolingLayer.feedForward(input);
    final Matrix error = maxPoolingLayer.backPropagate(input, expectedMaxPoolingActivation, nextError);
    assertTrue(expectedMaxPoolingError.compare(error, TOLERANCE));
  }

  @Test
  public void testAveragePoolingActivation() {
    final Matrix poolingActivation = averagePoolingLayer.feedForward(input);
    assertTrue(expectedAveragePoolingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testAveragePoolingBackPropagate() {
    final Matrix error = averagePoolingLayer.backPropagate(input, expectedAveragePoolingActivation, nextError);
    assertTrue(expectedAveragePoolingError.compare(error, TOLERANCE));
  }

  @Test
  public void testRemainderExistingMaxPoolingActivation() {
    final Matrix poolingActivation = remainderExistingMaxPoolingLayer.feedForward(input);
    assertTrue(expectedRemainderExistingMaxPoolingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testRemainderExistingMaxPoolingBackPropagate() {
    remainderExistingMaxPoolingLayer.feedForward(input);
    final Matrix error = remainderExistingMaxPoolingLayer
        .backPropagate(input, expectedRemainderExistingMaxPoolingActivation, nextError);
    assertTrue(expectedRemainderExistingMaxPoolingError.compare(error, TOLERANCE));
  }

  @Test
  public void testRemainderExistingAveragePoolingActivation() {
    final Matrix poolingActivation = remainderExistingAveragePoolingLayer.feedForward(input);
    assertTrue(expectedRemainderExistingAveragePoolingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testRemainderExistingAveragePoolingBackPropagate() {
    final Matrix error = remainderExistingAveragePoolingLayer
        .backPropagate(input, expectedRemainderExistingAveragePoolingActivation, nextError);
    assertTrue(expectedRemainderExistingAveragePoolingError.compare(error, TOLERANCE));
  }

  @Test
  public void testMaxPoolingWithPaddingActivation() {
    final Matrix poolingActivation = maxPoolingWithPaddingLayer.feedForward(input);
    assertTrue(expectedMaxPoolingWithPaddingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testMaxPoolingWithPaddingBackPropagate() {
    maxPoolingWithPaddingLayer.feedForward(input);
    final Matrix error = maxPoolingWithPaddingLayer
        .backPropagate(input, expectedMaxPoolingWithPaddingActivation, nextErrorWithPadding);
    assertTrue(expectedMaxPoolingWithPaddingError.compare(error, TOLERANCE));
  }

  @Test
  public void testAveragePoolingWithPaddingActivation() {
    final Matrix poolingActivation = averagePoolingWithPaddingLayer.feedForward(input);
    assertTrue(expectedAveragePoolingWithPaddingActivation.compare(poolingActivation, TOLERANCE));
  }

  @Test
  public void testAveragePoolingWithPaddingBackPropagate() {
    final Matrix error = averagePoolingWithPaddingLayer
        .backPropagate(input, expectedAveragePoolingWithPaddingActivation, nextErrorWithPadding);
    assertTrue(expectedAveragePoolingWithPaddingError.compare(error, TOLERANCE));
  }
}
