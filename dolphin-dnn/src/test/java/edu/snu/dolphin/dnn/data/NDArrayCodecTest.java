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

import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Test class for testing {@link NDArrayCodec}'s encoding and decoding functions.
 */
public final class NDArrayCodecTest {

  private NDArrayCodec ndArrayCodec;
  private Random random;

  @Before
  public void setUp() throws InjectionException {
    this.ndArrayCodec = Tang.Factory.getTang().newInjector().getInstance(NDArrayCodec.class);
    this.random = new Random();
  }

  /**
   * Checks that a random 2D NDArray does not change after encoding and decoding it, sequentially.
   */
  @Test
   public void testEncodeDecode2DArray() {
    final INDArray inputArray = NDArrayGenerator.generateRandomNDArray(random, 2);
    final INDArray retArray = ndArrayCodec.decode(ndArrayCodec.encode(inputArray));

    assertEquals(inputArray, retArray);
  }
  /**
   *
   * Checks that a random 3D NDArray does not change after encoding and decoding it, sequentially.
   */
  @Test
  public void testEncodeDecode3DArray() {
    final INDArray inputArray = NDArrayGenerator.generateRandomNDArray(random, 3);
    final INDArray retArray = ndArrayCodec.decode(ndArrayCodec.encode(inputArray));

    assertEquals("Encode-decode result is different from expected array", inputArray, retArray);
  }
}
