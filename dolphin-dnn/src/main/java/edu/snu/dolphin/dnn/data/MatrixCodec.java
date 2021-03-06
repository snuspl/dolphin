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
import org.apache.reef.io.network.impl.StreamingCodec;

import javax.inject.Inject;
import java.io.*;

/**
 * Serialization codec for {@link edu.snu.dolphin.dnn.blas.Matrix}.
 * Implements the {@code StreamingCodec} interface for efficient usages in other codec classes.
 */
public final class MatrixCodec implements StreamingCodec<Matrix> {

  private final MatrixFactory matrixFactory;

  @Inject
  private MatrixCodec(final MatrixFactory matrixFactory) {
    this.matrixFactory = matrixFactory;
  }

  @Override
  public byte[] encode(final Matrix matrix) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream(Integer.SIZE * 2 +
                                                                         Float.SIZE * matrix.getLength());
         final DataOutputStream dstream = new DataOutputStream(bstream)) {

      encodeToStream(matrix, dstream);
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during MatrixCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final Matrix matrix, final DataOutputStream dstream) {
    try {
      dstream.writeInt(matrix.getRows());
      dstream.writeInt(matrix.getColumns());

      for (int elementIndex = 0; elementIndex < matrix.getLength(); elementIndex++) {
        dstream.writeFloat(matrix.get(elementIndex));
      }

    } catch (final IOException e) {
      throw new RuntimeException("IOException during MatrixCodec.encodeToStream()", e);
    }
  }

  @Override
  public Matrix decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during MatrixCodec.decode()", e);
    }
  }

  @Override
  public Matrix decodeFromStream(final DataInputStream dstream) {
    try {
      final int rows = dstream.readInt();
      final int columns = dstream.readInt();
      final int length = rows * columns; // overflow may occur if matrix is too big, but in practice is not a concern

      final float[] elements = new float[length];
      for (int index = 0; index < length; index++) {
        elements[index] = dstream.readFloat();
      }

      return matrixFactory.create(elements, rows, columns);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during MatrixCodec.decodeFromStream()", e);
    }
  }
}
