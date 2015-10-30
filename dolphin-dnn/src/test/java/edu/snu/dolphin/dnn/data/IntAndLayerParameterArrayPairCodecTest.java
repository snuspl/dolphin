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

import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static edu.snu.dolphin.dnn.data.LayerParameterArrayCodecTest.generateRandomLayerParameterArray;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test class for testing {@link IntAndLayerParameterArrayPairCodec}'s encoding and decoding features.
 */
public final class IntAndLayerParameterArrayPairCodecTest {

  private IntAndLayerParameterArrayPairCodec intAndLayerParameterArrayPairCodec;
  private MatrixFactory matrixFactory;
  private final Random random = new Random();

  @Before
  public void setUp() throws InjectionException {

    final Configuration conf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();
    final Injector injector = Tang.Factory.getTang().newInjector(conf);

    this.intAndLayerParameterArrayPairCodec = injector.getInstance(IntAndLayerParameterArrayPairCodec.class);
    this.matrixFactory = injector.getInstance(MatrixFactory.class);
  }

  /**
   * Checks that a pair of an integer and an random array of layer parameters
   * does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeIntAndLayerParametersPair() {
    final Pair<Integer, LayerParameter[]> inputIntAndLayerParameterArrayPair =
        new Pair<>(random.nextInt(), generateRandomLayerParameterArray(matrixFactory, random, 10));
    final Pair<Integer, LayerParameter[]> outputIntAndLayerParameterArrayPair =
        intAndLayerParameterArrayPairCodec.decode(
            intAndLayerParameterArrayPairCodec.encode(inputIntAndLayerParameterArrayPair));

    assertEquals("Encode-decode result is different from the expected integer",
        inputIntAndLayerParameterArrayPair.getFirst(), outputIntAndLayerParameterArrayPair.getFirst());
    assertArrayEquals("Encode-decode result is different from the expected array of layer parameters",
        inputIntAndLayerParameterArrayPair.getSecond(), inputIntAndLayerParameterArrayPair.getSecond());
  }
}
