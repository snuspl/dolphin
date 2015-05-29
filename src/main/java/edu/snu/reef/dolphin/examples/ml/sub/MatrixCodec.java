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
package edu.snu.reef.dolphin.examples.ml.sub;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

/**
 * Codec for a matrix.
 */
public class MatrixCodec implements Codec<Matrix> {
  /**
   * This class is instantiated by TANG.
   */
  @Inject
  public MatrixCodec() {
    // Intentionally blank
  }

  @Override
  public Matrix decode(final byte[] data) {
    try (final DataInputStream dis = new DataInputStream(new ByteArrayInputStream(data))) {
      // read the size of matrix.
      final int rowSize = dis.readInt();
      final int columnSize = dis.readInt();
      final Matrix ret = new SparseMatrix(rowSize, columnSize);
      while (dis.available() > 0) {
        // read row index, column index, and value of each element.
        ret.setQuick(dis.readInt(), dis.readInt(), dis.readDouble());
      }
      return ret;
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }
  }

  @Override
  public byte[] encode(final Matrix matrix) {
    final int rowSize = matrix.rowSize(), columnSize = matrix.columnSize();
    try (final ByteArrayOutputStream baos = new ByteArrayOutputStream(rowSize * columnSize * Double.SIZE);
         final DataOutputStream dos = new DataOutputStream(baos)) {
      // tell the size of matrix.
      dos.writeInt(rowSize);
      dos.writeInt(columnSize);
      for (int i = 0; i < rowSize; ++i) {
        final Vector row = matrix.viewRow(i);
        for (int j = 0; j < columnSize; ++j) {
          final double value = row.get(j);
          if (value == 0) {
            continue;
          }
          // write down row index, column index, and value of each element.
          dos.writeInt(i);
          dos.writeInt(j);
          dos.writeDouble(value);
        }
      }
      return baos.toByteArray();
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }
  }
}