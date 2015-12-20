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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static edu.snu.dolphin.dnn.data.LayerParameterArrayCodecTest.generateRandomLayerParameterArray;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test class for testing {@link IntAndLayerParameterArrayPairListCodec}'s encoding and decoding features.
 */
public final class IntAndLayerParameterArrayPairListCodecTest {

  private IntAndLayerParameterArrayPairListCodec intAndLayerParameterArrayPairListCodec;
  private MatrixFactory matrixFactory;
  private final Random random = new Random();

  @Before
  public void setUp() throws InjectionException {

    final Configuration conf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();
    final Injector injector = Tang.Factory.getTang().newInjector(conf);

    this.intAndLayerParameterArrayPairListCodec = injector.getInstance(IntAndLayerParameterArrayPairListCodec.class);
    this.matrixFactory = injector.getInstance(MatrixFactory.class);
  }

  /**
   * Checks that a random list of pairs of an integer and an array of layer parameters
   * does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeIntAndLayerParametersPairList() {
    final List<Pair<Integer, LayerParameter[]>> inputIntAndLayerParameterArrayPairList = new ArrayList<>(5);
    for (int i = 0; i < inputIntAndLayerParameterArrayPairList.size(); ++i) {
      inputIntAndLayerParameterArrayPairList.add(
          new Pair<>(random.nextInt(), generateRandomLayerParameterArray(matrixFactory, random, 10)));
    }
    final List<Pair<Integer, LayerParameter[]>> outputIntAndLayerParameterArrayPairList =
        intAndLayerParameterArrayPairListCodec.decode(
            intAndLayerParameterArrayPairListCodec.encode(inputIntAndLayerParameterArrayPairList));

    assertArrayEquals(
        "Encode-decode result is different from the expected list of integer and layer parameter array pairs",
        inputIntAndLayerParameterArrayPairList.toArray(), outputIntAndLayerParameterArrayPairList.toArray());
  }
}
