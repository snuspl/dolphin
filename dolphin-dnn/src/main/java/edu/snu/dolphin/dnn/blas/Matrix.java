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
  float get(int i);

  /**
   * Returns elements specified by the linear indices.
   */
  Matrix get(int[] indices);

  /**
   * Returns a element specified by the row and column indices.
   */
  float get(int rowIndex, int columnIndex);

  /**
   * Sets a matrix element (linear indexing).
   */
  Matrix put(int i, float v);

  /**
   * Sets a matrix element.
   */
  Matrix put(int rowIndex, int columnIndex, float v);

  /**
   * Sets a column with the given column vector.
   */
  void putColumn(int i, Matrix v);

  /**
   * Sets a row with the given row vector.
   */
  void putRow(int i, Matrix v);

  /**
   * Returns a copy of a column.
   */
  Matrix getColumn(int i);

  /**
   * Returns a copy of a row.
   */
  Matrix getRow(int i);

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
  Matrix fill(float v);

  /**
   * Reshapes the matrix.
   * The number of elements must not change.
   */
  Matrix reshape(int newRows, int newColumns);

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
  Matrix copy(Matrix m);

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
  Matrix add(float v);

  /**
   * Adds a scalar (in place).
   */
  Matrix addi(float v);

  /**
   * Adds a matrix.
   */
  Matrix add(Matrix m);

  /**
   * Adds a matrix (in place).
   */
  Matrix addi(Matrix m);

  /**
   * Adds a vector to all columns of the matrix.
   */
  Matrix addColumnVector(Matrix v);

  /**
   * Adds a vector to all columns of the matrix (in place).
   */
  Matrix addiColumnVector(Matrix v);

  /**
   * Adds a vector to all rows of the matrix.
   */
  Matrix addRowVector(Matrix v);

  /**
   * Adds a vector to all rows of the matrix (in place).
   */
  Matrix addiRowVector(Matrix v);

  /**
   * Subtracts a scalar.
   */
  Matrix sub(float v);

  /**
   * Subtracts a scalar (in place).
   */
  Matrix subi(float v);

  /**
   * Subtracts a matrix.
   */
  Matrix sub(Matrix m);

  /**
   * Subtracts a matrix (in place).
   */
  Matrix subi(Matrix m);

  /**
   * Subtracts a vector to all columns of the matrix.
   */
  Matrix subColumnVector(Matrix v);

  /**
   * Subtracts a vector to all columns of the matrix (in place).
   */
  Matrix subiColumnVector(Matrix v);

  /**
   * Subtracts a vector to all rows of the matrix.
   */
  Matrix subRowVector(Matrix v);

  /**
   * Subtracts a vector to all rows of the matrix (in place).
   */
  Matrix subiRowVector(Matrix v);

  /**
   * (right-)subtracts a scalar.
   */
  Matrix rsub(float v);

  /**
   * (right-)subtracts a scalar (in place).
   */
  Matrix rsubi(float v);

  /**
   * (right-)subtracts a matrix.
   */
  Matrix rsub(Matrix m);

  /**
   * (right-)subtracts a matrix (in place).
   */
  Matrix rsubi(Matrix m);

  /**
   * Multiplies a scalar.
   */
  Matrix mul(float v);

  /**
   * Multiplies a scalar (in place).
   */
  Matrix muli(float v);

  /**
   * Multiplies a matrix.
   */
  Matrix mul(Matrix m);

  /**
   * Multiplies a matrix (in place).
   */
  Matrix muli(Matrix m);

  /**
   * Multiplies a vector to all columns of the matrix.
   */
  Matrix mulColumnVector(Matrix v);

  /**
   * Multiplies a vector to all columns of the matrix (in place).
   */
  Matrix muliColumnVector(Matrix v);

  /**
   * Multiplies a vector to all rows of the matrix.
   */
  Matrix mulRowVector(Matrix v);

  /**
   * Multiplies a vector to all rows of the matrix (in place).
   */
  Matrix muliRowVector(Matrix v);

  /**
   * Divides by a scalar.
   */
  Matrix div(float v);

  /**
   * Divides by a scalar (in place).
   */
  Matrix divi(float v);

  /**
   * Divides by a matrix.
   */
  Matrix div(Matrix m);

  /**
   * Divides by a matrix (in place).
   */
  Matrix divi(Matrix m);

  /**
   * Divides a vector to all columns of the matrix.
   */
  Matrix divColumnVector(Matrix v);

  /**
   * Divides a vector to all columns of the matrix (in place).
   */
  Matrix diviColumnVector(Matrix v);

  /**
   * Divides a vector to all rows of the matrix.
   */
  Matrix divRowVector(Matrix v);

  /**
   * Divides a vector to all rows of the matrix (in place).
   */
  Matrix diviRowVector(Matrix v);

  /**
   * (right-)divides by a scalar.
   */
  Matrix rdiv(float v);

  /**
   * (right-)divides by a scalar (in place).
   */
  Matrix rdivi(float v);

  /**
   * (right-)divides by a matrix.
   */
  Matrix rdiv(Matrix m);

  /**
   * (right-)divides by a matrix (in place).
   */
  Matrix rdivi(Matrix m);

  /**
   * Matrix-Matrix multiplication.
   */
  Matrix mmul(Matrix m);

  /**
   * Matrix-Matrix multiplication (in place).
   */
  Matrix mmuli(Matrix m);

  /**
   * Returns the maximum element of the matrix.
   */
  float max();

  /**
   * Returns column-wise maximums.
   */
  Matrix columnMaxs();

  /**
   * Returns row-wise maximums.
   */
  Matrix rowMaxs();

  /**
   * Returns the minimum element of the matrix.
   */
  float min();

  /**
   * Returns column-wise minimums.
   */
  Matrix columnMins();

  /**
   * Returns row-wise minimums.
   */
  Matrix rowMins();

  /**
   * Returns a row vector containing the sum of elements in each column.
   */
  Matrix columnSums();

  /**
   * Returns a column vector containing the sum of elements in each row.
   */
  Matrix rowSums();

  /**
   * Returns the sum of all elements in the matrix.
   */
  float sum();

  /**
   * Compares with the given matrix.
   *
   * Returns true if and only if the given matrix has the same size
   * and the maximal absolute difference in all elements is smaller than the specified tolerance.
   */
  boolean compare(Matrix m, float tolerance);
}
