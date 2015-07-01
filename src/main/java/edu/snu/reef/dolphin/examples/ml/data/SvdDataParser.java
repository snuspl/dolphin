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
package edu.snu.reef.dolphin.examples.ml.data;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.reef.io.data.loading.api.DataSet;
import org.apache.reef.io.network.util.Pair;
import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.ParseException;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Data parser for svd.
 */
public final class SvdDataParser implements DataParser<List<Triple<Integer, Integer, Double>>> {
  private static final Logger LOG = Logger.getLogger(SvdDataParser.class.getName());

  private final DataSet<LongWritable, Text> dataSet;
  private List<Triple<Integer, Integer, Double>> ret;
  private ParseException parseException;

  @Inject
  public SvdDataParser(final DataSet<LongWritable, Text> dataSet) {
    this.dataSet = dataSet;
  }

  @Override
  public List<Triple<Integer, Integer, Double>> get() throws ParseException {
    if (ret == null) {
      parse();
    }

    if (parseException != null) {
      throw parseException;
    }

    return ret;
  }

  @Override
  public void parse() {
    LOG.log(Level.INFO, "Parsing data started!");
    if (ret != null) {
      return;
    }

    ret = new LinkedList<>();
    for (final Pair<LongWritable, Text> keyValue : dataSet) {
      final String text = keyValue.second.toString();
      if (text.startsWith("#") || text.length() == 0) {
        continue;
      }

      final String[] split = text.trim().split("\\s+");
      if (split.length != 3) {
        parseException = new ParseException("Parsed failed: need row index, column index AND value");
        return;
      }

      final int rowIndex;
      final int columnIndex;
      try {
        rowIndex = Integer.parseInt(split[0]);
        columnIndex = Integer.parseInt(split[1]);
      } catch (final NumberFormatException e) {
        parseException = new ParseException("Parse failed: indices should be INT");
        return;
      }

      final double value;
      try {
        value = Double.parseDouble(split[2]);
      } catch (final NumberFormatException e) {
        parseException = new ParseException("Parse failed: value should be DOUBLE");
        return;
      }

      ret.add(new ImmutableTriple<>(rowIndex, columnIndex, value));
    }
    LOG.log(Level.INFO, ret.toString());
    LOG.log(Level.INFO, "Parsing data ended!");
  }
}