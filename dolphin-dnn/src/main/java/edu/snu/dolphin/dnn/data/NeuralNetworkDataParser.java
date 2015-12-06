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
package edu.snu.dolphin.dnn.data;

import edu.snu.dolphin.bsp.core.DataParser;
import edu.snu.dolphin.bsp.core.ParseException;
import edu.snu.dolphin.dnn.NeuralNetworkDriverParameters.Delimiter;
import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.reef.io.data.loading.api.DataSet;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static edu.snu.dolphin.dnn.blas.MatrixUtils.readNumpy;

/**
 * Data parser for neural network.
 *
 * Parses Numpy compatible plain text file.
 */
public final class NeuralNetworkDataParser implements DataParser<List<Pair<Pair<Matrix, Integer>, Boolean>>> {

  private final MatrixFactory matrixFactory;
  private final DataSet<LongWritable, Text> dataSet;
  private final String delimiter;
  private List<Pair<Pair<Matrix, Integer>, Boolean>> result;
  private ParseException parseException;

  @Inject
  private NeuralNetworkDataParser(final MatrixFactory matrixFactory,
                                  final DataSet<LongWritable, Text> dataSet,
                                  @Parameter(Delimiter.class)final String delimiter) {
    this.matrixFactory = matrixFactory;
    this.dataSet = dataSet;
    this.delimiter = delimiter;
  }

  /** {@inheritDoc} */
  @Override
  public List<Pair<Pair<Matrix, Integer>, Boolean>> get() throws ParseException {
    if (result == null) {
      parse();
    }
    if (parseException != null) {
      throw parseException;
    }
    return result;
  }

  /** {@inheritDoc} */
  @Override
  public void parse() {
    final List<Pair<Pair<Matrix, Integer>, Boolean>> trainingData = new ArrayList<>();

    for (final Pair<LongWritable, Text> keyValue : dataSet) {
      final String text = keyValue.getSecond().toString().trim();
      if (text.startsWith("#") || 0 == text.length()) {
        continue;
      }
      try {
        final Matrix input = readNumpy(matrixFactory, new ByteArrayInputStream(text.getBytes()), delimiter);
        final Matrix data = input.get(range(0, input.getColumns() - 2));
        final int label = (int) input.get(input.getColumns() - 2);
        final boolean isValidation = ((int) input.get(input.getColumns() - 1) == 1);
        trainingData.add(new Pair<>(new Pair<>(data, label), isValidation));
      } catch (final IOException e) {
        parseException = new ParseException("IOException: " + e.toString());
        return;
      }
    }
    result = trainingData;
  }

  /**
   * Generates an array of integers from begin (inclusive) to end (exclusive).
   *
   * @param begin the beginning value, inclusive
   * @param end the ending value, exclusive
   * @return a generated array.
   */
  private int[] range(final int begin, final int end) {
    if (begin > end) {
      throw new IllegalArgumentException("The beginning value should be less than or equal to the ending value");
    }
    final int num = end - begin;
    if (num == 0) {
      return new int[0];
    }
    final int[] ret = new int[num];
    for (int i = 0; i < num; ++i) {
      ret[i] = begin + i;
    }
    return ret;
  }
}
