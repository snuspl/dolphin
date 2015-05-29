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
import edu.snu.reef.dolphin.examples.ml.key.MatrixSigma;
import edu.snu.reef.dolphin.examples.ml.key.MatrixVT;
import edu.snu.reef.dolphin.examples.ml.parameters.ApproxCnt;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterSender;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Control task for computing eigen vectors and values used in svd.
 */
public class EigenCtrlTask extends UserControllerTask
    implements DataScatterSender<Integer>,
    DataBroadcastSender<Vector>,
    DataGatherReceiver<List<Pair<Integer, Double>>> {
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
   * One of SVD algorithm's output matrix Sigma.
   */
  private Matrix matrixSigma;

  /**
   * Transpose of one of SVD algorithm's output matrix V.
   */
  private Matrix matrixVT;

  /**
   * Vector used for computing column vector of matrix V.
   */
  private Vector vectorB;

  /**
   * The 2-norm of the vector b.
   */
  private double norm;

  /**
   * r is the maximum number of matrix A's singular values.
   */
  private int r;

  /**
   * Each compute task is responsible for specific rows scattered by controller task.
   */
  private List<Integer> targetRows;

  @Inject
  public EigenCtrlTask(final KeyValueStore keyValueStore,
                       @Parameter(ApproxCnt.class) final int approxCnt) {
    this.keyValueStore = keyValueStore;
    this.approxCnt = approxCnt;
  }

  @Override
  public void initialize() {
    // Stored by LoadMatrixStage
    LOG.log(Level.INFO, "Load the input matrix A and initialize the other members.");
    final Matrix matrixA = keyValueStore.get(MatrixA.class);
    matrixVT = matrixA.like(matrixA.columnSize(), matrixA.columnSize());
    matrixSigma = matrixA.like();
    vectorB = matrixVT.viewRow(0).clone().assign(1);
    r = Math.min(matrixA.rowSize(), matrixA.columnSize());
    targetRows = new LinkedList<>();
    for (int i = 0; i < matrixA.columnSize(); ++i) {
      targetRows.add(i);
    }
  }

  @Override
  public final void run(final int iteration) {
    if (iteration == 0) {
      return;
    } else if (iteration % (approxCnt + 1) != 0) {
      return;
    }

    // Deflation
    LOG.log(Level.INFO, "Deflation in iteration #" + iteration);
    // A step of power iteration is over.
    final int index = iteration / (approxCnt + 1) - 1;

    // The ith singular value is sqrt of the norm of ith column vector of matrix V.
    matrixSigma.setQuick(index, index, Math.sqrt(norm));
    matrixVT.assignRow(index, vectorB.clone().divide(norm));
  }

  @Override
  public final boolean isTerminated(final int iteration) {
    return iteration > r * (approxCnt + 1);
  }

  @Override
  public void cleanup() {
    LOG.log(Level.INFO, "Store the output matrix Sigma and VT");
    keyValueStore.put(MatrixSigma.class, matrixSigma);
    keyValueStore.put(MatrixVT.class, matrixVT);
  }

  @Override
  public Vector sendBroadcastData(final int iteration) {
    LOG.log(Level.INFO, "Sending column vector of matrix V in iteration #" + iteration);
    return vectorB;
  }

  @Override
  public void receiveGatherData(final int iteration, final List<List<Pair<Integer, Double>>> data) {
    if (iteration % (approxCnt + 1) == 0) {
      return;
    }

    // Power iteration
    LOG.log(Level.INFO, "Received column vector of matrix V in iteration #" + iteration);
    for (final List<Pair<Integer, Double>> list : data) {
      for (final Pair<Integer, Double> e : list) {
        vectorB.setQuick(e.getLeft(), e.getRight());
      }
    }

    norm = vectorB.norm(2);
  }

  @Override
  public List<Integer> sendScatterData(final int iteration) {
    if (iteration != 0) {
      return Collections.EMPTY_LIST;
    }

    LOG.log(Level.INFO, "Sending target rows in iteration #" + iteration);
    return targetRows;
  }
}