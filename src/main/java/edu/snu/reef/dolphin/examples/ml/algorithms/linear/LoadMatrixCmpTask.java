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
package edu.snu.reef.dolphin.examples.ml.algorithms.linear;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.ParseException;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.key.MatrixA;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherSender;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.mahout.math.Matrix;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Compute task for loading matrix.
 */
public final class LoadMatrixCmpTask extends UserComputeTask
    implements DataGatherSender<List<Triple<Integer, Integer, Double>>>,
    DataBroadcastReceiver<Matrix> {
  private static final Logger LOG = Logger.getLogger(LoadMatrixCmpTask.class.getName());

  /**
   * Parser object for input data that returns data assigned to this Task.
   */
  private final DataParser<List<Triple<Integer, Integer, Double>>> dataParser;

  /**
   * Storage for the input matrix A.
   */
  private final KeyValueStore keyValueStore;

  /**
   * Container for the partial input matrix.
   */
  private List<Triple<Integer, Integer, Double>> matrixValues;

  /**
   * Input matrix loaded by compute tasks and gathered by controller task.
   */
  private Matrix matrixA;

  @Inject
  public LoadMatrixCmpTask(final DataParser<List<Triple<Integer, Integer, Double>>> dataParser,
                           final KeyValueStore keyValueStore) {
    this.dataParser = dataParser;
    this.keyValueStore = keyValueStore;
  }

  @Override
  public void initialize() throws ParseException {
    // Read the partial input matrix
    LOG.log(Level.INFO, "Loading the partial input matrix.");
    this.matrixValues = this.dataParser.get();
  }

  @Override
  public void run(final int iteration) {
  }

  @Override
  public void cleanup() {
    // pass input matrix to the main process
    LOG.log(Level.INFO, "Store the input data for the main process.");
    keyValueStore.put(MatrixA.class, this.matrixA);
  }

  @Override
  public void receiveBroadcastData(final int iteration, final Matrix data) {
    if (iteration > 0) {
      // And receive the entire input matrix and target rows.
      LOG.log(Level.INFO, "Received the entire input matrix A in iteration #" + iteration);
      matrixA = data;
    }
  }

  @Override
  public List<Triple<Integer, Integer, Double>> sendGatherData(final int iteration) {
    // Send partial input matrix to controller task.
    LOG.log(Level.INFO, "Send the partial input matrix A in iteration #" + iteration);
    return matrixValues;
  }
}