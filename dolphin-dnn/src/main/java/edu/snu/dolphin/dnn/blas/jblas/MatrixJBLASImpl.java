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
package edu.snu.dolphin.dnn.blas.jblas;

import edu.snu.dolphin.dnn.blas.Matrix;
import org.jblas.FloatMatrix;

/**
 * Matrix implementation based on JBLAS.
 */
public class MatrixJBLASImpl implements Matrix {

  private final FloatMatrix matrix;

  MatrixJBLASImpl(final FloatMatrix matrix) {
    this.matrix = matrix;
  }

  @Override
  public int getRows() {
    return matrix.getRows();
  }

  @Override
  public int getColumns() {
    return matrix.getColumns();
  }

  @Override
  public float get(final int i) {
    return matrix.get(i);
  }

  @Override
  public Matrix get(final int[] indices) {
    return new MatrixJBLASImpl(matrix.get(indices));
  }

  @Override
  public float get(final int rowIndex, final int columnIndex) {
    return matrix.get(rowIndex, columnIndex);
  }

  @Override
  public Matrix put(final int i, final float v) {
    matrix.put(i, v);
    return this;
  }

  @Override
  public Matrix put(final int rowIndex, final int columnIndex, final float v) {
    matrix.put(rowIndex, columnIndex, v);
    return this;
  }

  @Override
  public void putColumn(final int i, final Matrix v) {
    checkImpl(v);
    matrix.putColumn(i, ((MatrixJBLASImpl) v).matrix);
  }

  @Override
  public void putRow(final int i, final Matrix v) {
    checkImpl(v);
    matrix.putRow(i, ((MatrixJBLASImpl) v).matrix);
  }

  @Override
  public Matrix getColumn(final int i) {
    return new MatrixJBLASImpl(matrix.getColumn(i));
  }

  @Override
  public Matrix getRow(final int i) {
    return new MatrixJBLASImpl(matrix.getRow(i));
  }

  @Override
  public int getLength() {
    return matrix.getLength();
  }

  @Override
  public boolean isColumnVector() {
    return matrix.isColumnVector();
  }

  @Override
  public boolean isRowVector() {
    return matrix.isRowVector();
  }

  @Override
  public Matrix fill(final float v) {
    matrix.fill(v);
    return this;
  }

  @Override
  public Matrix reshape(final int newRows, final int newColumns) {
    matrix.reshape(newRows, newColumns);
    return this;
  }

  @Override
  public String toString() {
    return matrix.toString();
  }

  @Override
  public float[] toFloatArray() {
    return matrix.toArray();
  }

  @Override
  public Matrix copy(final Matrix m) {
    checkImpl(m);
    matrix.copy(((MatrixJBLASImpl) m).matrix);
    return m;
  }

  @Override
  public Matrix dup() {
    return new MatrixJBLASImpl(matrix.dup());
  }

  @Override
  public Matrix transpose() {
    return new MatrixJBLASImpl(matrix.transpose());
  }

  @Override
  public Matrix add(final float v) {
    return new MatrixJBLASImpl(matrix.add(v));
  }

  @Override
  public Matrix addi(final float v) {
    matrix.addi(v);
    return this;
  }

  @Override
  public Matrix add(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.add(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix addi(final Matrix m) {
    checkImpl(m);
    matrix.addi(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix addColumnVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.addColumnVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix addiColumnVector(final Matrix v) {
    checkImpl(v);
    matrix.addiColumnVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix addRowVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.addRowVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix addiRowVector(final Matrix v) {
    checkImpl(v);
    matrix.addiRowVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix sub(final float v) {
    return new MatrixJBLASImpl(matrix.sub(v));
  }

  @Override
  public Matrix subi(final float v) {
    matrix.subi(v);
    return this;
  }

  @Override
  public Matrix sub(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.sub(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix subi(final Matrix m) {
    checkImpl(m);
    matrix.subi(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix subColumnVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.subColumnVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix subiColumnVector(final Matrix v) {
    checkImpl(v);
    matrix.subiColumnVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix subRowVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.subRowVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix subiRowVector(final Matrix v) {
    checkImpl(v);
    matrix.subiRowVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix rsub(final float v) {
    return new MatrixJBLASImpl(matrix.rsub(v));
  }

  @Override
  public Matrix rsubi(final float v) {
    matrix.rsubi(v);
    return this;
  }

  @Override
  public Matrix rsub(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.rsub(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix rsubi(final Matrix m) {
    checkImpl(m);
    matrix.rsubi(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix mul(final float v) {
    return new MatrixJBLASImpl(matrix.mul(v));
  }

  @Override
  public Matrix muli(final float v) {
    matrix.muli(v);
    return this;
  }

  @Override
  public Matrix mul(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.mul(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix muli(final Matrix m) {
    checkImpl(m);
    matrix.muli(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix mulColumnVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.mulColumnVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix muliColumnVector(final Matrix v) {
    checkImpl(v);
    matrix.muliColumnVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix mulRowVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.mulRowVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix muliRowVector(final Matrix v) {
    checkImpl(v);
    matrix.muliRowVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix div(final float v) {
    return new MatrixJBLASImpl(matrix.div(v));
  }

  @Override
  public Matrix divi(final float v) {
    matrix.divi(v);
    return this;
  }

  @Override
  public Matrix div(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.div(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix divi(final Matrix m) {
    checkImpl(m);
    matrix.divi(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix divColumnVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.divColumnVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix diviColumnVector(final Matrix v) {
    checkImpl(v);
    matrix.diviColumnVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix divRowVector(final Matrix v) {
    checkImpl(v);
    return new MatrixJBLASImpl(matrix.divRowVector(((MatrixJBLASImpl) v).matrix));
  }

  @Override
  public Matrix diviRowVector(final Matrix v) {
    checkImpl(v);
    matrix.diviRowVector(((MatrixJBLASImpl) v).matrix);
    return this;
  }

  @Override
  public Matrix rdiv(final float v) {
    return new MatrixJBLASImpl(matrix.rdiv(v));
  }

  @Override
  public Matrix rdivi(final float v) {
    matrix.rdivi(v);
    return this;
  }

  @Override
  public Matrix rdiv(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.rdiv(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix rdivi(final Matrix m) {
    checkImpl(m);
    matrix.rdivi(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public Matrix mmul(final Matrix m) {
    checkImpl(m);
    return new MatrixJBLASImpl(matrix.mmul(((MatrixJBLASImpl) m).matrix));
  }

  @Override
  public Matrix mmuli(final Matrix m) {
    checkImpl(m);
    matrix.mmuli(((MatrixJBLASImpl) m).matrix);
    return this;
  }

  @Override
  public boolean compare(final Matrix m, final float tolerance) {
    if (m instanceof MatrixJBLASImpl) {
      return matrix.compare(((MatrixJBLASImpl) m).matrix, tolerance);
    }
    return false;
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof MatrixJBLASImpl) {
      return matrix.equals(((MatrixJBLASImpl) o).matrix);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return matrix.hashCode();
  }

  public static MatrixJBLASImpl concatHorizontally(final MatrixJBLASImpl a, final MatrixJBLASImpl b) {
    return new MatrixJBLASImpl(FloatMatrix.concatHorizontally(a.matrix, b.matrix));
  }

  public static MatrixJBLASImpl concatVertically(final MatrixJBLASImpl a, final MatrixJBLASImpl b) {
    return new MatrixJBLASImpl(FloatMatrix.concatVertically(a.matrix, b.matrix));
  }

  private void checkImpl(final Matrix m) {
    if (!(m instanceof MatrixJBLASImpl)) {
      throw new IllegalArgumentException("The given matrix should be JBLAS based");
    }
  }
}
