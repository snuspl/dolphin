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
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Compute task before computing eigen vectors and values used in svd.
 */
public class PostEigenCmpTask extends UserComputeTask
    implements DataScatterReceiver<List<Integer>>,
    DataBroadcastReceiver<Double>,
    DataGatherSender<List<Triple<Integer, Integer, Double>>> {
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

  @Inject
  public PostEigenCmpTask(final KeyValueStore keyValueStore,
                          final OutputStreamProvider outputStreamProvider) {
    this.keyValueStore = keyValueStore;
    this.outputStreamProvider = outputStreamProvider;
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
  public void receiveBroadcastData(final int iteration, final Double data) {
    if (iteration == 0) {
      return;
    }

    // And assign it to ith column of matrix U after normalizing it.
    LOG.log(Level.INFO, "Received norm value of column vector in iteration #" + iteration);
    final Vector ui = matrixU.viewColumn(iteration - 1);
    ui.assign(ui.divide(data));
  }

  @Override
  public List<Triple<Integer, Integer, Double>> sendGatherData(final int iteration) {
    if (iteration < r) {
      LOG.log(Level.INFO, "Sending partial column vector of matrix U in iteration #" + iteration);
      final List<Triple<Integer, Integer, Double>> partialVec = new LinkedList<>();
      final Vector vi = matrixVT.viewRow(iteration);
      final Vector ui = matrixU.viewColumn(iteration);
      for (final int row : targetRows) {
        final double value = matrixA.viewRow(row).dot(vi);
        partialVec.add(new ImmutableTriple<>(0, row, value));
        ui.setQuick(row, value);
      }
      return partialVec;
    } else {
      // Compute the partial result matrix by multiplying partial matrix U, Sigma, and VT.
      LOG.log(Level.INFO, "Sending partial result matrix in iteration #" + iteration);
      final List<Triple<Integer, Integer, Double>> matrixValues = new LinkedList<>();
      for (final int row : targetRows) {
        final Vector uRow = matrixU.viewRow(row);
        final Vector uSigma = result.viewRow(row).clone();
        for (int i = 0; i < r; ++i) {
          uSigma.setQuick(i, uRow.get(i) * matrixSigma.get(i, i));
        }
        final Vector resultRow = result.viewRow(row);
        for (int i = 0; i < r; ++i) {
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