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
 * Interface for matrix whose elements are {@code float} values.
 *
 * Linear indexing is in column-major order.
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
  float get(int index);

  /**
   * Returns a row vector in which elements are specified by the given linear indices.
   * The modification on the returned vector does not affect this matrix.
   */
  Matrix get(int[] indices);

  /**
   * Returns a element specified by the row and column indices.
   */
  float get(int rowIndex, int columnIndex);

  /**
   * Sets a matrix element (linear indexing).
   */
  Matrix put(int index, float value);

  /**
   * Sets a matrix element.
   */
  Matrix put(int rowIndex, int columnIndex, float value);

  /**
   * Sets a column with the given column vector.
   */
  void putColumn(int index, Matrix vector);

  /**
   * Sets a row with the given row vector.
   */
  void putRow(int index, Matrix vector);

  /**
   * Returns a copy of a column.
   */
  Matrix getColumn(int index);

  /**
   * Returns a copy of a row.
   */
  Matrix getRow(int index);

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
  Matrix fill(float value);

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
   * Copies {@link Matrix} {@code matrix} to this.
   * @param matrix a source matrix
   * @return this matrix.
   */
  Matrix copy(Matrix matrix);

  /**
   * Returns a duplicate of this matrix.
   */
  Matrix dup();

  /**
   * Returns transposed copy of this matrix.
   */
  Matrix transpose();


  /* Operations */

  /**
   * Adds a scalar to all elements.
   */
  Matrix add(float value);

  /**
   * Adds a scalar to all elements (in place).
   */
  Matrix addi(float value);

  /**
   * Element-wise adds a matrix.
   */
  Matrix add(Matrix matrix);

  /**
   * Element-wise Adds a matrix (in place).
   */
  Matrix addi(Matrix matrix);

  /**
   * Element-wise adds a vector to all columns of the matrix.
   */
  Matrix addColumnVector(Matrix vector);

  /**
   * Element-wise adds a vector to all columns of the matrix (in place).
   */
  Matrix addiColumnVector(Matrix vector);

  /**
   * Element-wise adds a vector to all rows of the matrix.
   */
  Matrix addRowVector(Matrix vector);

  /**
   * Element-wise adds a vector to all rows of the matrix (in place).
   */
  Matrix addiRowVector(Matrix vector);

  /**
   * Subtracts a scalar to all elements.
   */
  Matrix sub(float value);

  /**
   * Subtracts a scalar to all elements (in place).
   */
  Matrix subi(float value);

  /**
   * Element-wise subtracts a matrix.
   */
  Matrix sub(Matrix matrix);

  /**
   * Element-wise subtracts a matrix (in place).
   */
  Matrix subi(Matrix matrix);

  /**
   * Element-wise subtracts a vector to all columns of the matrix.
   */
  Matrix subColumnVector(Matrix vector);

  /**
   * Element-wise subtracts a vector to all columns of the matrix (in place).
   */
  Matrix subiColumnVector(Matrix vector);

  /**
   * Element-wise subtracts a vector to all rows of the matrix.
   */
  Matrix subRowVector(Matrix vector);

  /**
   * Element-wise subtracts a vector to all rows of the matrix (in place).
   */
  Matrix subiRowVector(Matrix vector);

  /**
   * Element-wise (right-)subtracts a scalar to all elements.
   */
  Matrix rsub(float value);

  /**
   * Element-wise (right-)subtracts a scalar to all elements (in place).
   */
  Matrix rsubi(float value);

  /**
   * Element-wise (right-)subtracts a matrix.
   */
  Matrix rsub(Matrix matrix);

  /**
   * Element-wise (right-)subtracts a matrix (in place).
   */
  Matrix rsubi(Matrix matrix);

  /**
   * Multiplies a scalar to all elements.
   */
  Matrix mul(float value);

  /**
   * Multiplies a scalar to all elements (in place).
   */
  Matrix muli(float value);

  /**
   * Element-wise multiplies a matrix.
   */
  Matrix mul(Matrix matrix);

  /**
   * Element-wise multiplies a matrix (in place).
   */
  Matrix muli(Matrix matrix);

  /**
   * Element-wise multiplies a vector to all columns of the matrix.
   */
  Matrix mulColumnVector(Matrix vector);

  /**
   * Element-wise multiplies a vector to all columns of the matrix (in place).
   */
  Matrix muliColumnVector(Matrix vector);

  /**
   * Element-wise multiplies a vector to all rows of the matrix.
   */
  Matrix mulRowVector(Matrix vector);

  /**
   * Element-wise multiplies a vector to all rows of the matrix (in place).
   */
  Matrix muliRowVector(Matrix vector);

  /**
   * Divides by a scalar to all elements.
   */
  Matrix div(float value);

  /**
   * Divides by a scalar to all elements (in place).
   */
  Matrix divi(float value);

  /**
   * Element-wise divides by a matrix.
   */
  Matrix div(Matrix matrix);

  /**
   * Element-wise divides by a matrix (in place).
   */
  Matrix divi(Matrix matrix);

  /**
   * Element-wise divides a vector to all columns of the matrix.
   */
  Matrix divColumnVector(Matrix vector);

  /**
   * Element-wise divides a vector to all columns of the matrix (in place).
   */
  Matrix diviColumnVector(Matrix vector);

  /**
   * Element-wise divides a vector to all rows of the matrix.
   */
  Matrix divRowVector(Matrix vector);

  /**
   * Element-wise divides a vector to all rows of the matrix (in place).
   */
  Matrix diviRowVector(Matrix vector);

  /**
   * Element-wise (right-)divides by a scalar to all elements.
   */
  Matrix rdiv(float value);

  /**
   * Element-wise (right-)divides by a scalar to all elements (in place).
   */
  Matrix rdivi(float value);

  /**
   * Element-wise (right-)divides by a matrix.
   */
  Matrix rdiv(Matrix matrix);

  /**
   * Element-wise (right-)divides by a matrix (in place).
   */
  Matrix rdivi(Matrix matrix);

  /**
   * Matrix-Matrix multiplication.
   */
  Matrix mmul(Matrix matrix);

  /**
   * Matrix-Matrix multiplication (in place).
   */
  Matrix mmuli(Matrix matrix);

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
  boolean compare(Matrix matrix, float tolerance);
}
