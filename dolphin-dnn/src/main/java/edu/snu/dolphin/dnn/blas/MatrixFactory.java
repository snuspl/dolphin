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
 * Factory interface for {@link Matrix}.
 */
public interface MatrixFactory {

  /**
   * Creates a row vector.
   * @param length the length of a row vector
   * @return a generated row vector
   */
  Matrix create(final int length);

  /**
   * Creates a matrix.
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix create(final int rows, final int columns);

  /**
   * Creates a row vector with the given values.
   * @param data elements of a row vector
   * @return a generated row vector
   */
  Matrix create(final float[] data);

  /**
   * Creates a matrix with the given values.
   * @param data elements of a matrix
   * @return a generated matrix
   */
  Matrix create(final float[][] data);

  /**
   * Creates a matrix with the given values.
   * @param data elements of a matrix in column-major order
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix create(final float[] data, final int rows, final int columns);

  /**
   * Creates a row vector in which all elements are equal to {@code 1}.
   * @param length the length of a row vector
   * @return a generated row vector
   */
  Matrix ones(final int length);

  /**
   * Creates a matrix in which all elements are equal to {@code 1}.
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix ones(final int rows, final int columns);

  /**
   * Creates a row vector in which all elements are equal to {@code 0}.
   * @param length the length of a row vector
   * @return a generated row vector
   */
  Matrix zeros(final int length);

  /**
   * Creates a matrix in which all elements are equal to {@code 0}.
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix zeros(final int rows, final int columns);

  /**
   * Creates a row vector with random values uniformly distributed in 0..1.
   * @param length the length of a row vector
   * @return a generated row vector
   */
  Matrix rand(final int length);

  /**
   * Creates a matrix with random values uniformly distributed in 0..1.
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix rand(final int rows, final int columns);

  /**
   * Creates a matrix with random values uniformly distributed in 0..1.
   * @param rows the number of rows
   * @param columns the number of columns
   * @param seed a random seed
   * @return a generated matrix
   */
  Matrix rand(final int rows, final int columns, long seed);

  /**
   * Creates a row vector with normally distributed random values.
   * @param length the length of a row vector
   * @return a generated row vector
   */
  Matrix randn(final int length);

  /**
   * Creates a matrix with normally distributed random values.
   * @param rows the number of rows
   * @param columns the number of columns
   * @return a generated matrix
   */
  Matrix randn(final int rows, final int columns);

  /**
   * Creates a matrix with normally distributed random values.
   * @param rows the number of rows
   * @param columns the number of columns
   * @param seed a random seed
   * @return a generated matrix
   */
  Matrix randn(final int rows, final int columns, long seed);

  /**
   * Concatenates two matrices horizontally.
   */
  Matrix concatHorizontally(Matrix a, Matrix b);

  /**
   * Concatenates two matrices vertically.
   */
  Matrix concatVertically(Matrix a, Matrix b);
}
