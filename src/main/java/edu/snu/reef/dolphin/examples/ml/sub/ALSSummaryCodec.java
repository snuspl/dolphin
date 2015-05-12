/**
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

import edu.snu.reef.dolphin.examples.ml.data.ALSSummary;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

public final class ALSSummaryCodec implements Codec<ALSSummary> {

  @Inject
  public ALSSummaryCodec() {
  }

  @Override
  public final byte[] encode(final ALSSummary alsSummary) {
    Matrix matrix = alsSummary.getMatrix();
    final ByteArrayOutputStream baos =
        new ByteArrayOutputStream(Double.SIZE * matrix.columnSize() * matrix.rowSize()
                                  + Integer.SIZE * 3);

    try (final DataOutputStream daos = new DataOutputStream(baos)) {
      daos.writeBoolean(alsSummary.getUserItem() == ALSSummary.UserItem.USER);
      daos.writeInt(matrix.rowSize());
      daos.writeInt(matrix.columnSize());

      for (int i = 0; i < matrix.rowSize(); i++) {
        for (int j = 0; j < matrix.columnSize(); j++) {
          daos.writeDouble(matrix.get(i, j));
        }
      }
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return baos.toByteArray();
  }

  @Override
  public final ALSSummary decode(final byte[] data) {
    final ByteArrayInputStream bais = new ByteArrayInputStream(data);
    final ALSSummary alsSummary;

    try (final DataInputStream dais = new DataInputStream(bais)) {
      final ALSSummary.UserItem userItem =
          dais.readBoolean() ? ALSSummary.UserItem.USER : ALSSummary.UserItem.ITEM;
      final int rowSize = dais.readInt();
      final int colSize = dais.readInt();

      Matrix matrix = new DenseMatrix(rowSize, colSize);
      for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
          matrix.set(i, j, dais.readDouble());
        }
      }
      alsSummary = new ALSSummary(matrix, userItem);

    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return alsSummary;
  }
}
