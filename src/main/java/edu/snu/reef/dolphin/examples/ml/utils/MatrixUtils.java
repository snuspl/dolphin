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
package edu.snu.reef.dolphin.examples.ml.utils;

import org.apache.mahout.math.Matrix;

import java.util.Collection;

public final class MatrixUtils {
  public final static Matrix viewColumns(final Matrix matrix, final Collection<Integer> indices) {
    Matrix retMatrix = matrix.like(matrix.rowSize(), indices.size());

    int retMatrixIndex = 0;
    for (final int matrixIndex : indices) {
      retMatrix.assignColumn(retMatrixIndex++, matrix.viewColumn(matrixIndex));
    }

    return retMatrix;
  }
}
