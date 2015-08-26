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

import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Test class for testing {@link ActivationGradientListCodec}'s encoding and decoding functions.
 */
public final class ActivationGradientListCodecTest {

  private ActivationGradientListCodec activationGradientListCodec;
  private Random random;

  @Before
  public void setUp() throws InjectionException {
    this.activationGradientListCodec =
        Tang.Factory.getTang().newInjector().getInstance(ActivationGradientListCodec.class);
    this.random = new Random();
  }

  /**
   * Checks that a random set of activations and gradients does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecodeActivationGradient() {
    final List<Pair<List<INDArray>, List<INDArray>>> inputList = new ArrayList<>(10);
    for (int index = 0; index < inputList.size(); index++) {
      final List<INDArray> activation = new ArrayList<>(10);
      for (int activationIndex = 0; activationIndex < activation.size(); activationIndex++) {
        activation.add(NDArrayGenerator.generateRandomNDArray(random, 2));
      }

      final List<INDArray> gradient = new ArrayList<>(10);
      for (int gradientIndex = 0; gradientIndex < gradient.size(); gradientIndex++) {
        gradient.add(NDArrayGenerator.generateRandomNDArray(random, 2));
      }

      inputList.add(new Pair<>(activation, gradient));
    }

    final List<Pair<List<INDArray>, List<INDArray>>> retList =
        activationGradientListCodec.decode(activationGradientListCodec.encode(inputList));

    assertEquals("Encode-decode result is different from expected list", inputList, retList);
  }
}
