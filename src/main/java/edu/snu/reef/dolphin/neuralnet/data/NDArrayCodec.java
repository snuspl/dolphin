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

import org.apache.reef.io.network.impl.StreamingCodec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.io.*;

/**
 * Serialization codec for ND4J `NDArray`s.
 * Implements the {@code StreamingCodec} interface for efficient usages in other codec classes.
 */
public final class NDArrayCodec implements StreamingCodec<INDArray> {

  @Inject
  private NDArrayCodec() {
  }

  @Override
  public byte[] encode(final INDArray ndArray) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream(Integer.SIZE * (ndArray.shape().length + 1) +
                                                                         Float.SIZE * ndArray.length());
         final DataOutputStream dstream = new DataOutputStream(bstream)) {

      encodeToStream(ndArray, dstream);
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NDArrayCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final INDArray ndArray, final DataOutputStream dstream) {
    try {
      final int[] shape = ndArray.shape();
      dstream.writeInt(shape.length);

      int elementCount = 1;
      for (final int dimension : shape) {
        dstream.writeInt(dimension);
        elementCount *= dimension; // overflow may occur if ndArray is too big, but in practice is not a concern
      }

      for (int elementIndex = 0; elementIndex < elementCount; elementIndex++) {
        dstream.writeFloat(ndArray.getFloat(elementIndex));
      }

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NDArrayCodec.encodeToStream()", e);
    }
  }

  @Override
  public INDArray decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NDArrayCodec.decode()", e);
    }
  }

  @Override
  public INDArray decodeFromStream(final DataInputStream dstream) {
    try {
      final int[] shape = new int[dstream.readInt()];

      int elementCount = 1;
      for (int shapeIndex = 0; shapeIndex < shape.length; shapeIndex++) {
        final int dimension = dstream.readInt();
        shape[shapeIndex] = dimension;
        elementCount *= dimension; // overflow may occur if ndArray is too big, but in practice is not a concern
      }

      final float[] elements = new float[elementCount];
      for (int index = 0; index < elementCount; index++) {
        elements[index] = dstream.readFloat();
      }

      return Nd4j.create(elements, shape);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NDArrayCodec.decodeFromStream()", e);
    }
  }
}
