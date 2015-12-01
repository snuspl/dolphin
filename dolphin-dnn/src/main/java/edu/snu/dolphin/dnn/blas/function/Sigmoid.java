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
package edu.snu.dolphin.dnn.blas.function;

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFunctions;

/**
 * Sigmoid function.
 */
final class Sigmoid implements Function {

  /**
   * Applies a sigmoid function to all elements of the specified matrix.
   */
  @Override
  public Matrix apply(final Matrix m) {
    return applyi(m.dup());
  }

  /**
   * Applies a sigmoid function to all elements of the specified matrix (in place).
   */
  @Override
  public Matrix applyi(final Matrix m) {
    return MatrixFunctions.expi(MatrixFunctions.negi(m)).addi(1.0f).rdivi(1.0f);
  }

  /**
   * Calculates the matrix in which all elements are derivatives of a sigmoid function.
   */
  @Override
  public Matrix derivative(final Matrix m) {
    return derivativei(m.dup());
  }

  /**
   * Calculates the matrix in which all elements are derivatives of a sigmoid function (in place).
   */
  @Override
  public Matrix derivativei(final Matrix m) {
    applyi(m);
    return m.muli(m.rsub(1.0f));
  }
}
