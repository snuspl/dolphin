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
package edu.snu.reef.dolphin.neuralnet.data;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.ParseException;
import edu.snu.reef.dolphin.neuralnet.NeuralNetworkDriverParameters;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.reef.io.data.loading.api.DataSet;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.inject.Inject;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static edu.snu.reef.dolphin.neuralnet.util.Nd4jUtils.readNumpy;

/**
 * Data parser for neural network.
 *
 * Parses Numpy compatible plain text file.
 */
public final class NeuralNetworkDataParser implements DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> {

  private final DataSet<LongWritable, Text> dataSet;
  private final String delimiter;
  private List<Pair<Pair<INDArray, Integer>, Boolean>> result;
  private ParseException parseException;

  @Inject
  public NeuralNetworkDataParser(final DataSet<LongWritable, Text> dataSet,
                                 @Parameter(NeuralNetworkDriverParameters.Delimiter.class) final String delimiter) {
    this.dataSet = dataSet;
    this.delimiter = delimiter;
  }

  /** {@inheritDoc} */
  @Override
  public List<Pair<Pair<INDArray, Integer>, Boolean>> get() throws ParseException {
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
    final List<Pair<Pair<INDArray, Integer>, Boolean>> trainingData = new ArrayList<>();

    for (final Pair<LongWritable, Text> keyValue : dataSet) {
      try {
        final INDArray input = readNumpy(
            new ByteArrayInputStream(keyValue.getSecond().toString().getBytes()), delimiter);
        final INDArray data = input.get(NDArrayIndex.interval(0, input.columns() - 2));
        final int label = (int) input.getDouble(input.columns() - 2);
        final boolean isValidation = ((int) input.getDouble(input.columns() - 1) == 1);
        trainingData.add(new Pair<>(new Pair<>(data, label), isValidation));
      } catch (final IOException e) {
        parseException = new ParseException("IOException: " + e.toString());
        return;
      }
    }
    result = trainingData;
  }
}
