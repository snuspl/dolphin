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
package edu.snu.reef.dolphin.neuralnet.function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Sigmoid Function.
 */
public final class Sigmoid {

  private Sigmoid() {
  }

  /**
   * Returns the output of sigmoid.
   * @param x
   * @return
   */
  public static double value(final double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  /**
   * Applies sigmoid to each element of the matrix.
   * @param m
   * @return
   */
  public static DoubleMatrix value(final DoubleMatrix m) {
    final DoubleMatrix ret = m.dup();
    valuei(ret);
    return ret;
  }

  /**
   * Applies sigmoid to each element of the input matrix. (in-place)
   * @param m
   */
  public static void valuei(final DoubleMatrix m) {
    MatrixFunctions.expi(m.negi()).addi(1.0).rdivi(1.0);
  }

  /**
   * Returns the derivative of sigmoid.
   * @param a the output of sigmoid.
   * @return
   */
  public static double derivative(final double a) {
    return a * (1 - a);
  }

  /**
   * Applies the derivative of sigmoid to each element of matrix.
   * @param a the matrix of which each element is sigmoid output.
   * @return
   */
  public static DoubleMatrix derivative(final DoubleMatrix a) {
    final DoubleMatrix ret = a.dup();
    derivativei(ret);
    return ret;
  }

  /**
   * Applies the derivative of sigmoid to each element of matrix. (in-place)
   * @param a the matrix of which each element is sigmoid output.
   */
  public static void derivativei(final DoubleMatrix a) {
    a.muli(a.rsub(1.0));
  }
}
