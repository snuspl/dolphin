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
import edu.snu.reef.dolphin.core.OutputStreamProvider;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.key.MatrixA;
import edu.snu.reef.dolphin.examples.ml.key.MatrixSigma;
import edu.snu.reef.dolphin.examples.ml.key.MatrixVT;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterSender;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import javax.inject.Inject;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Control task after computing eigen vectors and values used in svd.
 */
public class PostEigenCtrlTask extends UserControllerTask
    implements DataScatterSender<Integer>,
    DataBroadcastSender<List<Double>>,
    DataGatherReceiver<List<Triple<Integer, Integer, Double>>> {
  private static final String LINE_SEPARATOR = System.getProperty("line.separator");

  private static final Logger LOG = Logger.getLogger(PostEigenCtrlTask.class.getName());

  /**
   * Storage for the matrix A, Sigma, and VT.
   */
  private final KeyValueStore keyValueStore;

  /**
   * Output service provider to logging matrix A, U, Sigma, VT and result.
   */
  private final OutputStreamProvider outputStreamProvider;

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

  /**
   * The 2-norm of the column vector of matrix U.
   */
  private List<Double> norms;

  @Inject
  public PostEigenCtrlTask(final KeyValueStore keyValueStore,
                           final OutputStreamProvider outputStreamProvider) {
    this.keyValueStore = keyValueStore;
    this.outputStreamProvider = outputStreamProvider;
  }

  @Override
  public void initialize() {
    LOG.log(Level.INFO, "Load the matrix A, Sigma, U and initialize the other members.");
    matrixA = this.keyValueStore.get(MatrixA.class);
    matrixU = matrixA.like(matrixA.rowSize(), matrixA.rowSize());
    matrixSigma = this.keyValueStore.get(MatrixSigma.class);
    matrixVT = this.keyValueStore.get(MatrixVT.class);
    result = matrixA.like();
    r = Math.min(matrixA.rowSize(), matrixA.columnSize());
    targetRows = new LinkedList<>();
    for (int i = 0; i < matrixA.rowSize(); ++i) {
      targetRows.add(i);
    }
    norms = new LinkedList<>();
  }

  @Override
  public final void run(final int iteration) {
  }

  @Override
  public final boolean isTerminated(final int iteration) {
    return iteration > 1;
  }

  @Override
  public void cleanup() {
    LOG.log(Level.INFO, "Print out matrix A, U, Sigma, VT, result");
    final StringBuilder sb = new StringBuilder();
    appendMatrix(sb, "A", matrixA);
    appendMatrix(sb, "U", matrixU);
    appendMatrix(sb, "Sigma", matrixSigma);
    appendMatrix(sb, "VT", matrixVT);
    appendMatrix(sb, "result", result);
    try (final DataOutputStream dataOutputStream = outputStreamProvider.create("SVD")) {
      dataOutputStream.writeBytes(sb.toString());
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public List<Double> sendBroadcastData(final int iteration) {
    if (iteration != 1) {
      return Collections.EMPTY_LIST;
    }

    LOG.log(Level.INFO, "Sending norm value of column vector of matrix U in iteration #" + iteration);
    return norms;
  }

  @Override
  public void receiveGatherData(final int iteration, final List<List<Triple<Integer, Integer, Double>>> data) {
    if (iteration == 0) {
      // U column vector gather
      LOG.log(Level.INFO, "Received the column vector of matrix U in iteration #" + iteration);

      for (final List<Triple<Integer, Integer, Double>> list : data) {
        for (final Triple<Integer, Integer, Double> datum : list) {
          matrixU.setQuick(datum.getLeft(), datum.getMiddle(), datum.getRight());
        }
      }

      for (int i = 0; i < r; ++i) {
        // Compute the norm of entire ith column vector of matrix U and broadcast it.
        Vector ui = matrixU.viewColumn(i);
        double norm = ui.norm(2);
        norms.add(norm);
        ui.assign(ui.divide(norm));
      }
    } else if (iteration == 1) {
      // result matrix gather
      LOG.log(Level.INFO, "Received the result matrix in iteration #" + iteration);
      for (final List<Triple<Integer, Integer, Double>> list : data) {
        for (final Triple<Integer, Integer, Double> datum : list) {
          result.setQuick(datum.getLeft(), datum.getMiddle(), datum.getRight());
        }
      }
    }
  }

  @Override
  public List<Integer> sendScatterData(final int iteration) {
    if (iteration != 0) {
      return Collections.EMPTY_LIST;
    }

    LOG.log(Level.INFO, "Sending target rows in iteration #" + iteration);
    return targetRows;
  }

  private void appendMatrix(final StringBuilder sb, final String matrixName, final Matrix matrix) {
    sb.append(matrixName).append(" : ").append(LINE_SEPARATOR);

    final int rowSize = matrix.rowSize();
    for (int i = 0; i < rowSize; ++i) {
      final Vector row = matrix.viewRow(i);

      if (row.getNumNonZeroElements() == 0) {
        continue;
      }

      sb.append(i).append(" : ").append(row).append(LINE_SEPARATOR);
    }
  }
}