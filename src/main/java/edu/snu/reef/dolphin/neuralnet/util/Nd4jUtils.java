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
package edu.snu.reef.dolphin.neuralnet.util;

import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Utility class for ND4J library.
 */
public final class Nd4jUtils {

  private Nd4jUtils() {
  }

  /**
   * Returns true if each element of one matrix is equal to one of another within tolerance.
   * @param a one matrix to be tested for equality.
   * @param b another matrix to be tested for equality.
   * @param tolerance the maximum difference for which both numbers are still considered equal.
   * @return true if each element of one matrix is equal to one of another.
   */
  public static boolean equals(final INDArray a, final INDArray b, final float tolerance) {
    if (!Arrays.equals(a.shape(), b.shape())) {
      return false;
    }
    for (int i = 0; i < a.rows(); ++i) {
      for (int j = 0; j < a.columns(); ++j) {
        if (Math.abs(a.getFloat(i, j) - b.getFloat(i, j)) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns true if the two specified matrix lists are equal to one another within tolerance.
   * @param a one matrix list to be tested for equality.
   * @param b another matrix list to be tested for equality.
   * @param tolerance the maximum difference for which both numbers are still considered equal.
   * @return true if the two specified matrix lists are equal to one another.
   */
  public static boolean equals(final List<INDArray> a, final List<INDArray> b, final float tolerance) {
    if (a.size() != b.size()) {
      return false;
    }
    final Iterator bIter = b.iterator();
    for (final INDArray m : a) {
      if (!equals(m, (INDArray) bIter.next(), tolerance)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns true if each element of weight and bias of a layer parameter is equal to another within tolerance.
   * @param a one layer parameter array to be tested for equality.
   * @param b another layer parameter array to be tested for equality.
   * @param tolerance the maximum difference for which both numbers are still considered equal.
   * @return true if two layer parameter arrays are equal.
   */
  public static boolean equals(final LayerParameter[] a, final LayerParameter[] b, final float tolerance) {
    if (a.length != b.length) {
      return false;
    }
    for (int i = 0; i < a.length; ++i) {
      final LayerParameter param = a[i];
      final LayerParameter other = b[i];
      if (!Nd4jUtils.equals(param.getBiasParam(), other.getBiasParam(), tolerance)
          || !Nd4jUtils.equals(param.getWeightParam(), other.getWeightParam(), tolerance)) {
        return false;
      }
    }
    return true;
  }

  public static INDArray readNumpy(final InputStream filePath, final String split) throws IOException {
    final BufferedReader reader = new BufferedReader(new InputStreamReader(filePath));
    String line;
    final List<float[]> data2 = new ArrayList<>();
    int numColumns = -1;
    INDArray ret;
    while ((line = reader.readLine()) != null) {
      final String[] data = line.trim().split(split);
      if (numColumns < 0) {
        numColumns = data.length;

      } else {
        assert data.length == numColumns : "Data has inconsistent number of columns";
      }
      data2.add(readSplit(data));
    }

    ret = Nd4j.create(data2.size(), numColumns);
    for (int i = 0; i < data2.size(); i++) {
      ret.putRow(i, Nd4j.create(Nd4j.createBuffer(data2.get(i))));
    }
    return ret;
  }

  private static float[] readSplit(final String[] split) {
    final float[] ret = new float[split.length];
    for (int i = 0; i < split.length; i++) {
      ret[i] = Float.parseFloat(split[i]);
    }
    return ret;
  }

  public static INDArray readNumpy(final String filePath, final String split) throws IOException {
    return readNumpy(new FileInputStream(filePath), split);
  }
}
