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
package edu.snu.reef.dolphin.neuralnet.data;

import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static edu.snu.reef.dolphin.neuralnet.data.LayerParameterArrayCodecTest.generateRandomLayerParameterArray;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test class for testing {@link LayerParameterArrayListCodec}'s encoding and decoding features.
 */
public final class LayerParameterArrayListCodecTest {

  private LayerParameterArrayListCodec layerParameterArrayListCodec;

  @Before
  public void setUp() throws InjectionException {
    this.layerParameterArrayListCodec =
        Tang.Factory.getTang().newInjector().getInstance(LayerParameterArrayListCodec.class);
  }

  /**
   * Checks that a random list of layer parameter arrays does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeLayerParameters() {
    final List<LayerParameter[]> inputLayerParameterArrayList = new ArrayList<>(5);
    for (int i = 0; i < inputLayerParameterArrayList.size(); ++i) {
      inputLayerParameterArrayList.add(generateRandomLayerParameterArray());
    }
    final List<LayerParameter[]> outputLayerParameterArrayList =
        layerParameterArrayListCodec.decode(layerParameterArrayListCodec.encode(inputLayerParameterArrayList));

    assertArrayEquals("Encode-decode result is different from the expected list of LayerParameter arrays",
        inputLayerParameterArrayList.toArray(), outputLayerParameterArrayList.toArray());
  }
}
