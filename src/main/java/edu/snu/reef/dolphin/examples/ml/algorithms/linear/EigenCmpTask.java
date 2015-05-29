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
import edu.snu.reef.dolphin.examples.ml.parameters.ApproxCnt;
import edu.snu.reef.dolphin.groupcomm.interfaces.*;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Compute task for computing eigen vectors and values used in svd.
 */
public class EigenCmpTask extends UserComputeTask
    implements DataScatterReceiver<List<Integer>>,
    DataBroadcastReceiver<Vector>,
    DataGatherSender<List<Pair<Integer, Double>>> {
  private static final Logger LOG = Logger.getLogger(EigenCtrlTask.class.getName());

  /**
   * Storage for the matrix A, Sigma, and VT.
   */
  private final KeyValueStore keyValueStore;

  /**
   * The number of power iteration.
   */
  private final int approxCnt;

  /**
   * The input matrix A.
   */
  private Matrix matrixA;

  /**
   * One of SVD algorithm's output matrix Sigma.
   */
  private Matrix matrixSigma;

  /**
   * Transpose of one of SVD algorithm's output matrix V.
   */
  private Matrix matrixVT;

  /**
   * The result of matrix multiplication partial matrix AT(transpose of A) and A.
   */
  private Matrix matrixATA;

  /**
   * Each compute task is responsible for specific rows scattered by controller task.
   */
  private List<Integer> targetRows;

  /**
   * Partial vector used for computing column vector of matrix V.
   */
  private List<Pair<Integer, Double>> partialVec;

  /**
   * Received vector from EigenCtrlTask.
   */
  private Vector dataVector;

  /**
   * State types in eigen compute task.
   */
  private enum State {
    RECEIVE_TARGET_ROWS, POWER_ITERATION, DEFLATION
  }

  /**
   * State in eigen compute task assigned in receiveBroadcastData.
   */
  private State state;

  @Inject
  public EigenCmpTask(final KeyValueStore keyValueStore,
                      @Parameter(ApproxCnt.class) final int approxCnt) {
    this.keyValueStore = keyValueStore;
    this.approxCnt = approxCnt;
  }

  @Override
  public void initialize() throws ParseException {
    LOG.log(Level.INFO, "Load the input matrix A and initialize the other members.");
    matrixA = keyValueStore.get(MatrixA.class);
    matrixSigma = matrixA.like();
    matrixVT = matrixA.like(matrixA.columnSize(), matrixA.columnSize());
    matrixATA = matrixA.like(matrixA.columnSize(), matrixA.columnSize());
    targetRows = new LinkedList<>();
    partialVec = new LinkedList<>();
  }

  @Override
  public final void run(final int iteration) {
    if (state == State.DEFLATION) {
      // Deflation should
      // Remove the eigen value and the eigen vector from matrix ATA using hotelling deflation.
      LOG.log(Level.INFO, "Received column vector of V in iteration #" + iteration);
      final int index = iteration / (approxCnt + 1) - 1;
      final double norm = dataVector.norm(2);
      final Vector vectorB = dataVector.divide(norm);
      matrixVT.assignRow(index, vectorB);

      // Each compute task is responsible for specific rows
      for (final int row : targetRows) {
        final Vector targetRow = matrixATA.viewRow(row);
        targetRow.assign(targetRow.minus(vectorB.times(vectorB.get(row)).times(norm)));
      }

      // Store the singular value from controller task.
      matrixSigma.setQuick(index, index, Math.sqrt(norm));
    } else if (state == State.POWER_ITERATION) {
      // Power iteration
      // Multiply matrix ATA and a vector from controller task to compute the column vector of V
      // Each compute task is responsible for specific rows and sends the partial column vector of V
      LOG.log(Level.INFO, "Received column vector of V in iteration #" + iteration);
      final double norm = dataVector.norm(2);
      final Vector vectorB = dataVector.divide(norm);

      partialVec.clear();
      for (final int row : targetRows) {
        final double value = matrixATA.viewRow(row).dot(vectorB);
        partialVec.add(new ImmutablePair<>(row, value));
      }
    }
  }

  @Override
  public void cleanup() {
    LOG.log(Level.INFO, "Store the output matrix Sigma and VT");
    keyValueStore.put(MatrixSigma.class, matrixSigma);
    keyValueStore.put(MatrixVT.class, matrixVT);
  }

  @Override
  public void receiveBroadcastData(final int iteration, final Vector data) {
    if (iteration == 0) {
      state = State.RECEIVE_TARGET_ROWS;
    } else if (iteration % (approxCnt + 1) == 0) {
      state = State.DEFLATION;
    } else {
      state = State.POWER_ITERATION;
    }

    // Used in run method.
    dataVector = data;
  }

  @Override
  public List<Pair<Integer, Double>> sendGatherData(final int iteration) {
    LOG.log(Level.INFO, "Sending partial column vector of matrix V in iteration #" + iteration);
    return partialVec;
  }

  @Override
  public void receiveScatterData(final int iteration, final List<Integer> data) {
    if (state != State.RECEIVE_TARGET_ROWS) {
      return;
    } else if (data == null) {
      return;
    }

    // Compute matrix ATA(AT is transpose of matrix A).
    // Because the column vectors of matrix V are the eigen vectors of ATA.
    LOG.log(Level.INFO, "Received target rows in iteration #" + iteration);
    targetRows.addAll(data);
    for (final int row : targetRows) {
      final Vector targetRow = matrixATA.viewRow(row);
      final Vector column = matrixA.viewColumn(row);
      for (int i = 0; i < matrixA.columnSize(); ++i) {
        targetRow.setQuick(i, column.dot(matrixA.viewColumn(i)));
      }
    }
  }
}