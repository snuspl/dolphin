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
package edu.snu.dolphin.dnn.data;

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test class for testing {@link LayerParameterArrayCodec}'s encoding and decoding features.
 */
public final class LayerParameterArrayCodecTest {

  private LayerParameterArrayCodec layerParameterArrayCodec;
  private MatrixFactory matrixFactory;

  @Before
  public void setUp() throws InjectionException {
    final Configuration conf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();
    final Injector injector = Tang.Factory.getTang().newInjector(conf);

    this.layerParameterArrayCodec = injector.getInstance(LayerParameterArrayCodec.class);
    this.matrixFactory = injector.getInstance(MatrixFactory.class);
  }

  /**
   * Checks that a random layer parameter array does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeLayerParameters() {
    final LayerParameter[] inputLayerParameters = generateRandomLayerParameterArray(matrixFactory);
    final LayerParameter[] outputLayerParameters =
        layerParameterArrayCodec.decode(layerParameterArrayCodec.encode(inputLayerParameters));

    assertArrayEquals("Encode-decode result is different from expected LayerParameter array",
        inputLayerParameters, outputLayerParameters);
  }

  /**
   * @return a random 10-element array of layer parameters.
   */
  public static LayerParameter[] generateRandomLayerParameterArray(final MatrixFactory matrixFactory) {
    return generateRandomLayerParameterArray(matrixFactory, new Random(), 10);
  }

  /**
   * @param random a random number generator.
   * @param numElements the number of array elements
   * @return a random array of layer parameters.
   */
  public static LayerParameter[] generateRandomLayerParameterArray(final MatrixFactory matrixFactory,
                                                                   final Random random,
                                                                   final int numElements) {
    final LayerParameter[] retLayerParameters = new LayerParameter[numElements];
    for (int index = 0; index < retLayerParameters.length; index++) {
      final Matrix weightParam = MatrixGenerator.generateRandomMatrix(matrixFactory, random);
      final Matrix biasParam = MatrixGenerator.generateRandomMatrix(matrixFactory, random);
      retLayerParameters[index] = LayerParameter.newBuilder()
          .setWeightParam(weightParam)
          .setBiasParam(biasParam)
          .build();
    }
    return retLayerParameters;
  }
}
