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

/**
 * Function interface.
 */
public interface Function {

  Matrix apply(final Matrix m);

  Matrix applyi(final Matrix m);

  Matrix derivative(final Matrix m);

  Matrix derivativei(final Matrix m);

  final class Factory {

    private Factory() {
    }

    public static Function getFunction(final String name) {
      switch (name.toLowerCase()) {
      case "sigmoid":
        return new Sigmoid();
      case "relu":
        return new ReLU();
      default:
        throw new IllegalArgumentException("Unsupported function: " + name);
      }
    }
  }
}
