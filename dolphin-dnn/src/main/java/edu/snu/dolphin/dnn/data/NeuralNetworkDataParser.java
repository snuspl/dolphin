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
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.BatchSize;
import org.apache.commons.lang.ArrayUtils;
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
public final class NeuralNetworkDataParser implements DataParser<List<Pair<Pair<Matrix, int[]>, Boolean>>> {

  private final MatrixFactory matrixFactory;
  private final DataSet<LongWritable, Text> dataSet;
  private final String delimiter;
  private final int batchSize;
  private List<Pair<Pair<Matrix, int[]>, Boolean>> result;
  private ParseException parseException;

  @Inject
  private NeuralNetworkDataParser(final MatrixFactory matrixFactory,
                                  final DataSet<LongWritable, Text> dataSet,
                                  @Parameter(Delimiter.class)final String delimiter,
                                  @Parameter(BatchSize.class) final int batchSize) {
    this.matrixFactory = matrixFactory;
    this.dataSet = dataSet;
    this.delimiter = delimiter;
    this.batchSize = batchSize;
  }

  /** {@inheritDoc} */
  @Override
  public List<Pair<Pair<Matrix, int[]>, Boolean>> get() throws ParseException {
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
    final List<Pair<Pair<Matrix, int[]>, Boolean>> dataList = new ArrayList<>();
    final BatchGenerator trainingBatchGenerator = new BatchGenerator(matrixFactory, batchSize);
    final BatchGenerator validationBatchGenerator = new BatchGenerator(matrixFactory, batchSize);

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

        if (isValidation) {
          final Pair<Matrix, int[]> batch = validationBatchGenerator.push(data, label);
          if (batch != null) {
            dataList.add(new Pair<>(batch, true));
          }
        } else {
          final Pair<Matrix, int[]> batch = trainingBatchGenerator.push(data, label);
          if (batch != null) {
            dataList.add(new Pair<>(batch, false));
          }
        }
      } catch (final IOException e) {
        parseException = new ParseException("IOException: " + e.toString());
        return;
      }
    }

    if (validationBatchGenerator.size() > 0) {
      dataList.add(new Pair<>(validationBatchGenerator.pull(), false));
    }

    if (trainingBatchGenerator.size() > 0) {
      dataList.add(new Pair<>(trainingBatchGenerator.pull(), false));
    }

    result = dataList;
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

  /**
   * Class for generating batch matrix and an array of labels with the specified batch size.
   */
  public static final class BatchGenerator {
    private final MatrixFactory matrixFactory;
    private final int batchSize;
    private final List<Matrix> dataList;
    private final List<Integer> labelList;

    public BatchGenerator(final MatrixFactory matrixFactory, final int batchSize) {
      this.matrixFactory = matrixFactory;
      this.batchSize = batchSize;
      this.dataList = new ArrayList<>(batchSize);
      this.labelList = new ArrayList<>(batchSize);
    }

    /**
     * @return the number of aggregated data.
     */
    public int size() {
      return dataList.size();
    }

    /**
     * Pushes a data and label.
     * @param data a single datum
     * @param label a label for the datum.
     * @return a pair of a batch input matrix and an array of labels, if data have been pushed
     *         with the specified batch size. or {@code null}, otherwise.
     */
    public Pair<Matrix, int[]> push(final Matrix data, final int label) {
      dataList.add(data);
      labelList.add(label);

      if (dataList.size() == batchSize) {
        return pull();
      } else {
        return null;
      }
    }

    /**
     * @return a pair of a batch input matrix and an array of labels that have been pushed.
     */
    public Pair<Matrix, int[]> pull() {
      final Pair<Matrix, int[]> ret = new Pair<>(makeBatch(dataList),
          ArrayUtils.toPrimitive(labelList.toArray(new Integer[labelList.size()])));
      dataList.clear();
      labelList.clear();
      return ret;
    }

    /**
     * Generates a batch input matrix with the specified list of input data.
     * @param inputs a list of input data
     * @return a batch input matrix
     */
    private Matrix makeBatch(final List<Matrix> inputs) {
      if (inputs.size() > 0) {
        final Matrix ret = matrixFactory.create(inputs.size(), inputs.get(0).getLength());
        int i = 0;
        for (final Matrix vector : inputs) {
          ret.putRow(i++, vector);
        }
        return ret;
      } else {
        throw new IllegalArgumentException("At least one ndarray is needed to make batch");
      }
    }
  }
}
