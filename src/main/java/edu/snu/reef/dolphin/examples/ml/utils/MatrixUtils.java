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
