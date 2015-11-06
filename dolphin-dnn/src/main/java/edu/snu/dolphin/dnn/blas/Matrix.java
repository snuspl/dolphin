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
package edu.snu.dolphin.dnn.blas;

/**
 * Matrix interface.
 */
public interface Matrix {

  /**
   * Returns the number of rows.
   */
  int getRows();

  /**
   * Returns the number of columns.
   */
  int getColumns();

  /**
   * Returns a element specified by the given index (linear indexing).
   */
  float get(final int i);

  /**
   * Returns elements specified by the linear indices.
   */
  Matrix get(final int[] indices);

  /**
   * Returns a element specified by the row and column indices.
   */
  float get(final int rowIndex, final int columnIndex);

  /**
   * Sets a matrix element (linear indexing).
   */
  Matrix put(final int i, final float v);

  /**
   * Sets a matrix element.
   */
  Matrix put(final int rowIndex, final int columnIndex, final float v);

  /**
   * Sets a column with the given column vector.
   */
  void putColumn(final int i, Matrix v);

  /**
   * Sets a row with the given row vector.
   */
  void putRow(final int i, Matrix v);

  /**
   * Returns a copy of a column.
   */
  Matrix getColumn(final int i);

  /**
   * Returns a copy of a row.
   */
  Matrix getRow(final int i);

  /**
   * Returns total number of elements.
   */
  int getLength();

  /**
   * Checks whether the matrix is a column vector.
   */
  boolean isColumnVector();

  /**
   * Checks whether the matrix is a row vector.
   */
  boolean isRowVector();

  /**
   * Sets all elements to the specified value.
   */
  Matrix fill(final float v);

  /**
   * Reshapes the matrix.
   * The number of elements must not change.
   */
  Matrix reshape(final int newRows, final int newColumns);

  /**
   * Returns a string representation of this matrix.
   */
  String toString();

  /**
   * Converts the matrix to a one-dimensional array of {@code float}s.
   */
  float[] toFloatArray();

  /**
   * Copies {@code Matrix} {@code m} to this.
   * @param m a source matrix
   * @return a source matrix {@code m}
   */
  Matrix copy(final Matrix m);

  /**
   * @return a duplicate of this matrix.
   */
  Matrix dup();

  /**
   * Returns transposed copy of this matrix.
   */
  Matrix transpose();


  /* Operations */

  /**
   * Adds a scalar.
   */
  Matrix add(final float v);

  /**
   * Adds a scalar (in place).
   */
  Matrix addi(final float v);

  /**
   * Adds a matrix.
   */
  Matrix add(final Matrix m);

  /**
   * Adds a matrix (in place).
   */
  Matrix addi(final Matrix m);

  /**
   * Adds a vector to all columns of the matrix.
   */
  Matrix addColumnVector(final Matrix v);

  /**
   * Adds a vector to all columns of the matrix (in place).
   */
  Matrix addiColumnVector(final Matrix v);

  /**
   * Adds a vector to all rows of the matrix.
   */
  Matrix addRowVector(final Matrix v);

  /**
   * Adds a vector to all rows of the matrix (in place).
   */
  Matrix addiRowVector(final Matrix v);

  /**
   * Subtracts a scalar.
   */
  Matrix sub(final float v);

  /**
   * Subtracts a scalar (in place).
   */
  Matrix subi(final float v);

  /**
   * Subtracts a matrix.
   */
  Matrix sub(final Matrix m);

  /**
   * Subtracts a matrix (in place).
   */
  Matrix subi(final Matrix m);

  /**
   * Subtracts a vector to all columns of the matrix.
   */
  Matrix subColumnVector(final Matrix v);

  /**
   * Subtracts a vector to all columns of the matrix (in place).
   */
  Matrix subiColumnVector(final Matrix v);

  /**
   * Subtracts a vector to all rows of the matrix.
   */
  Matrix subRowVector(final Matrix v);

  /**
   * Subtracts a vector to all rows of the matrix (in place).
   */
  Matrix subiRowVector(final Matrix v);

  /**
   * (right-)subtracts a scalar.
   */
  Matrix rsub(final float v);

  /**
   * (right-)subtracts a scalar (in place).
   */
  Matrix rsubi(final float v);

  /**
   * (right-)subtracts a matrix.
   */
  Matrix rsub(final Matrix m);

  /**
   * (right-)subtracts a matrix (in place).
   */
  Matrix rsubi(final Matrix m);

  /**
   * Multiplies a scalar.
   */
  Matrix mul(final float v);

  /**
   * Multiplies a scalar (in place).
   */
  Matrix muli(final float v);

  /**
   * Multiplies a matrix.
   */
  Matrix mul(final Matrix m);

  /**
   * Multiplies a matrix (in place).
   */
  Matrix muli(final Matrix m);

  /**
   * Multiplies a vector to all columns of the matrix.
   */
  Matrix mulColumnVector(final Matrix v);

  /**
   * Multiplies a vector to all columns of the matrix (in place).
   */
  Matrix muliColumnVector(final Matrix v);

  /**
   * Multiplies a vector to all rows of the matrix.
   */
  Matrix mulRowVector(final Matrix v);

  /**
   * Multiplies a vector to all rows of the matrix (in place).
   */
  Matrix muliRowVector(final Matrix v);

  /**
   * Divides by a scalar.
   */
  Matrix div(final float v);

  /**
   * Divides by a scalar (in place).
   */
  Matrix divi(final float v);

  /**
   * Divides by a matrix.
   */
  Matrix div(final Matrix m);

  /**
   * Divides by a matrix (in place).
   */
  Matrix divi(final Matrix m);

  /**
   * Divides a vector to all columns of the matrix.
   */
  Matrix divColumnVector(final Matrix v);

  /**
   * Divides a vector to all columns of the matrix (in place).
   */
  Matrix diviColumnVector(final Matrix v);

  /**
   * Divides a vector to all rows of the matrix.
   */
  Matrix divRowVector(final Matrix v);

  /**
   * Divides a vector to all rows of the matrix (in place).
   */
  Matrix diviRowVector(final Matrix v);

  /**
   * (right-)divides by a scalar.
   */
  Matrix rdiv(final float v);

  /**
   * (right-)divides by a scalar (in place).
   */
  Matrix rdivi(final float v);

  /**
   * (right-)divides by a matrix.
   */
  Matrix rdiv(final Matrix m);

  /**
   * (right-)divides by a matrix (in place).
   */
  Matrix rdivi(final Matrix m);

  /**
   * Matrix-Matrix multiplication.
   */
  Matrix mmul(final Matrix m);

  /**
   * Matrix-Matrix multiplication (in place).
   */
  Matrix mmuli(final Matrix m);

  /**
   * Compares with the given matrix.
   *
   * Returns true if and only if the given matrix has the same size
   * and the maximal absolute difference in all elements is smaller than the specified tolerance.
   */
  boolean compare(final Matrix m, float tolerance);
}
