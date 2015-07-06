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
import edu.snu.reef.dolphin.core.ParseException;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.key.MatrixA;
import edu.snu.reef.dolphin.examples.ml.key.MatrixSigma;
import edu.snu.reef.dolphin.examples.ml.key.MatrixVT;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterReceiver;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Compute task after computing eigen vectors and values used in svd.
 */
public class PostEigenCmpTask extends UserComputeTask
    implements DataScatterReceiver<List<Integer>>,
    DataBroadcastReceiver<List<Double>>,
    DataGatherSender<List<Triple<Integer, Integer, Double>>> {
  private static final Logger LOG = Logger.getLogger(PostEigenCtrlTask.class.getName());

  /**
   * Storage for the matrix A, Sigma, and VT.
   */
  private final KeyValueStore keyValueStore;

  /**
   * The input matrix A.
   */
  private Matrix matrixA;

  /**
   * One of SVD algorithm's output matrix U.
   */
  private Matrix matrixU;

  /**
   * One of SVD algorithm's output matrix Sigma.
   */
  private Matrix matrixSigma;

  /**
   * Transpose of one of SVD algorithm's output matrix V.
   */
  private Matrix matrixVT;

  /**
   * Matrix result is U * Sigma * VT(transpose of V).
   */
  private Matrix result;

  /**
   * r is the maximum number of matrix A's singular values.
   */
  private int r;

  /**
   * Each compute task is responsible for specific rows scattered by controller task.
   */
  private List<Integer> targetRows;

  @Inject
  public PostEigenCmpTask(final KeyValueStore keyValueStore) {
    this.keyValueStore = keyValueStore;
  }

  @Override
  public void initialize() throws ParseException {
    LOG.log(Level.INFO, "Load the matrix A, Sigma, U and initialize the other members.");
    matrixA = this.keyValueStore.get(MatrixA.class);
    r = Math.min(matrixA.rowSize(), matrixA.columnSize());
    matrixU = matrixA.like(matrixA.rowSize(), matrixA.rowSize());
    matrixSigma = this.keyValueStore.get(MatrixSigma.class);
    matrixVT = this.keyValueStore.get(MatrixVT.class);
    result = matrixA.like();
    targetRows = new LinkedList<>();
  }

  @Override
  public final void run(final int iteration) {
  }

  @Override
  public void cleanup() {
  }

  @Override
  public void receiveBroadcastData(final int iteration, final List<Double> data) {
    if (iteration != 1) {
      return;
    } else if (data == null) {
      return;
    }

    // And assign it to ith column of matrix U after normalizing it.
    LOG.log(Level.INFO, "Received norm value of column vector in iteration #" + iteration);

    for (int i = 0; i < data.size(); ++i) {
      final Vector ui = matrixU.viewColumn(i);
      ui.assign(ui.divide(data.get(i)));
    }
  }

  @Override
  public List<Triple<Integer, Integer, Double>> sendGatherData(final int iteration) {
    if (iteration == 0) {
      LOG.log(Level.INFO, "Sending partial column vectors of matrix U in iteration #" + iteration);
      final List<Triple<Integer, Integer, Double>> partialVectors = new LinkedList<>();

      for (int i = 0; i < r; ++i) {
        final Vector vi = matrixVT.viewRow(i);
        final Vector ui = matrixU.viewColumn(i);

        for (final int row : targetRows) {
          final double value = matrixA.viewRow(row).dot(vi);
          partialVectors.add(new ImmutableTriple<>(row, i, value));
          ui.setQuick(row, value);
        }
      }

      return partialVectors;
    } else {
      // Compute the partial result matrix by multiplying partial matrix U, Sigma, and VT.
      LOG.log(Level.INFO, "Sending partial result matrix in iteration #" + iteration);
      final List<Triple<Integer, Integer, Double>> matrixValues = new LinkedList<>();
      for (final int row : targetRows) {
        final Vector uRow = matrixU.viewRow(row);
        final Vector uSigma = result.viewRow(row).clone();
        for (int i = 0; i < uRow.size(); ++i) {
          uSigma.setQuick(i, uRow.get(i) * matrixSigma.get(i, i));
        }
        final Vector resultRow = result.viewRow(row);
        for (int i = 0; i < matrixVT.columnSize(); ++i) {
          final double value = uSigma.dot(matrixVT.viewColumn(i));
          resultRow.setQuick(i, value);
          matrixValues.add(new ImmutableTriple<>(row, i, value));
        }
      }
      return matrixValues;
    }
  }

  @Override
  public void receiveScatterData(final int iteration, final List<Integer> data) {
    if (iteration > 0) {
      return;
    } else if (data == null) {
      return;
    }

    LOG.log(Level.INFO, "Received target rows in iteration #" + iteration);
    targetRows = data;
  }
}