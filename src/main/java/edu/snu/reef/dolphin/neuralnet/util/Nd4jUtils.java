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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for ND4J library.
 */
public final class Nd4jUtils {

  private Nd4jUtils() {
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
