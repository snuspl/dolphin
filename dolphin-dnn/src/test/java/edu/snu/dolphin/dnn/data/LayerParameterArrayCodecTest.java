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

import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test class for testing {@link LayerParameterArrayCodec}'s encoding and decoding features.
 */
public final class LayerParameterArrayCodecTest {

  private LayerParameterArrayCodec layerParameterArrayCodec;

  @Before
  public void setUp() throws InjectionException {
    this.layerParameterArrayCodec = Tang.Factory.getTang().newInjector().getInstance(LayerParameterArrayCodec.class);
  }

  /**
   * Checks that a random layer parameter array does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeLayerParameters() {
    final LayerParameter[] inputLayerParameters = generateRandomLayerParameterArray();
    final LayerParameter[] outputLayerParameters =
        layerParameterArrayCodec.decode(layerParameterArrayCodec.encode(inputLayerParameters));

    assertArrayEquals("Encode-decode result is different from expected LayerParameter array",
        inputLayerParameters, outputLayerParameters);
  }

  /**
   * @return a random 10-element array of layer parameters.
   */
  public static LayerParameter[] generateRandomLayerParameterArray() {
    return generateRandomLayerParameterArray(new Random(), 10);
  }

  /**
   * @param random a random number generator.
   * @param numElements the number of array elements
   * @return a random array of layer parameters.
   */
  public static LayerParameter[] generateRandomLayerParameterArray(final Random random,
                                                                   final int numElements) {
    final LayerParameter[] retLayerParameters = new LayerParameter[numElements];
    for (int index = 0; index < retLayerParameters.length; index++) {
      final INDArray weightParam = NDArrayGenerator.generateRandomNDArray(random, 2);
      final INDArray biasParam = NDArrayGenerator.generateRandomNDArray(random, 2);
      retLayerParameters[index] = LayerParameter.newBuilder()
          .setWeightParam(weightParam)
          .setBiasParam(biasParam)
          .build();
    }
    return retLayerParameters;
  }
}
