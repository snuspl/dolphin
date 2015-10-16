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

import org.apache.reef.io.network.util.Pair;
import org.apache.reef.io.serialization.Codec;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Serialization codec for neural network activations and error values.
 * Assumes activations and errors are a list of {@code INDArray}s.
 * Internally uses {@link NDArrayCodec}.
 */
public final class ActivationErrorListCodec implements Codec<List<Pair<List<INDArray>, List<INDArray>>>> {

  private final NDArrayCodec ndArrayCodec;

  @Inject
  private ActivationErrorListCodec(final NDArrayCodec ndArrayCodec) {
    this.ndArrayCodec = ndArrayCodec;
  }

  @Override
  public byte[] encode(final List<Pair<List<INDArray>, List<INDArray>>> activationsErrorsList) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {

      dstream.writeInt(activationsErrorsList.size());
      for (final Pair<List<INDArray>, List<INDArray>> activationError : activationsErrorsList) {
        final List<INDArray> activations = activationError.getFirst();
        dstream.writeInt(activations.size());
        for (final INDArray activation : activations) {
          ndArrayCodec.encodeToStream(activation, dstream);
        }

        final List<INDArray> errors = activationError.getSecond();
        dstream.writeInt(errors.size());
        for (final INDArray error : errors) {
          ndArrayCodec.encodeToStream(error, dstream);
        }
      }

      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException while encoding activationsErrorsList", e);
    }
  }

  @Override
  public List<Pair<List<INDArray>, List<INDArray>>> decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {

      final int outerListSize = dstream.readInt();
      final List<Pair<List<INDArray>, List<INDArray>>> retList = new ArrayList<>(outerListSize);

      for (int outerIndex = 0; outerIndex < outerListSize; outerIndex++) {
        final int activationCount = dstream.readInt();
        final List<INDArray> activations = new ArrayList<>(activationCount);
        for (int activationIndex = 0; activationIndex < activationCount; activationIndex++) {
          activations.add(ndArrayCodec.decodeFromStream(dstream));
        }

        final int errorCount = dstream.readInt();
        final List<INDArray> errors = new ArrayList<>(errorCount);
        for (int errorIndex = 0; errorIndex < errorCount; errorIndex++) {
          errors.add(ndArrayCodec.decodeFromStream(dstream));
        }

        retList.add(new Pair<>(activations, errors));
      }

      return retList;

    } catch (final IOException e) {
      throw new RuntimeException("IOException while decoding activationsErrorsList", e);
    }
  }
}
