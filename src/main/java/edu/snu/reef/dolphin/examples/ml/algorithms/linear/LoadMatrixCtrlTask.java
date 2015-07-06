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

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.key.MatrixA;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

final class LoadMatrixCtrlTask extends UserControllerTask
    implements DataGatherReceiver<List<Triple<Integer, Integer, Double>>>,
    DataBroadcastSender<Matrix> {
  private static final Logger LOG = Logger.getLogger(LoadMatrixCtrlTask.class.getName());

  /**
   * Storage for the input matrix A.
   */
  private final KeyValueStore keyValueStore;

  /**
   * Input matrix loaded by compute tasks and gathered by controller task.
   */
  private Matrix matrixA;

  @Inject
  public LoadMatrixCtrlTask(final KeyValueStore keyValueStore) {
    LOG.log(Level.INFO, "Started.");
    this.keyValueStore = keyValueStore;
    // Assigning matrix A is essential for broadcasting it.
    // If A is null, broadcasting would be failed.
    this.matrixA = new SparseMatrix(0, 0);
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
  public boolean isTerminated(final int iteration) {
    // iteration == 0 : gather the partial input matrix
    // iteration == 1 : broadcast the entire input matrix
    return iteration > 1;
  }

  @Override
  public Matrix sendBroadcastData(final int iteration) {
    // broadcast the entire input matrix to compute tasks.
    LOG.log(Level.INFO, "Matrix A in iteration #" + iteration);
    return matrixA;
  }

  @Override
  public void receiveGatherData(final int iteration, final List<List<Triple<Integer, Integer, Double>>> data) {
    if (iteration > 0) {
      return;
    }

    LOG.log(Level.INFO, "Received partial matrix in iteration #" + iteration);

    int rows = 0, columns = 0;
    for (final List<Triple<Integer, Integer, Double>> list : data) {
      for (final Triple<Integer, Integer, Double> datum : list) {
        rows = Math.max(rows, datum.getLeft());
        columns = Math.max(columns, datum.getMiddle());
      }
    }

    // The number of input matrix's rows is greater than the max row index by 1.
    // The columns are the same with the rows.
    matrixA = matrixA.like(rows + 1, columns + 1);
    for (final List<Triple<Integer, Integer, Double>> list : data) {
      for (final Triple<Integer, Integer, Double> datum : list) {
        matrixA.setQuick(datum.getLeft(), datum.getMiddle(), datum.getRight());
      }
    }
  }
}