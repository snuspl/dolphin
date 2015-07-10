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
package edu.snu.reef.dolphin.examples.ml.algorithms.network;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.ParseException;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.data.NeuralNetworkModel;
import edu.snu.reef.dolphin.examples.ml.data.Row;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterReceiver;

import javax.inject.Inject;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.abs;
import static java.lang.Math.exp;

public class MultilayerPerceptronCmpTask extends UserComputeTask
    implements DataBroadcastReceiver<List<Double>>,
    DataScatterReceiver<List<Integer>>,
    DataGatherSender<List<Double>> {
  private final DataParser<List<Row>> dataParser;
  private List<Row> rows;
  private NeuralNetworkModel model;
  private boolean feedForward;
  private int step;
  private final Random random;
  private final double eta;
  private final double momentum;
  private List<List<Integer>> responsibleUnits;

  @Inject
  public MultilayerPerceptronCmpTask(final DataParser<List<Row>> dataParser) {
    this.dataParser = dataParser;
    this.random = new Random();
    this.eta = 0.3;
    this.momentum = 0.3;
  }

  @Override
  public void initialize() throws ParseException {
    this.rows = dataParser.get();
  }

  @Override
  public void receiveBroadcastData(final int iteration, final List<Double> data) {
    if (iteration == 0) {
      // the same model with control task
      final List<Integer> nUnits = new ArrayList<>(data.size());

      for (final Double i : data) {
        nUnits.add(i.intValue());
      }

      this.model = new NeuralNetworkModel(nUnits);

      // for vertical partitioning
      this.responsibleUnits = new ArrayList<>(this.model.getNumOfLayers());
    }

    // half is feed forward and the remain is back propagation.
    this.feedForward = iteration % (this.model.getNumOfWeights() * 2) < this.model.getNumOfWeights();

    // In feed forward or back propagation's step
    this.step = iteration % this.model.getNumOfWeights();

    if (this.step == 0) {
      // ignore input unit's activation in feed forward and output layer's errors in back propagation
      return;
    }

    if (this.feedForward) {
      // feed forward
      this.model.setUnits(this.step, data);
    } else {
      // back propagation
      this.model.setDelta(this.model.getNumOfDelta() - this.step, data);
    }
  }

  @Override
  public void receiveScatterData(final int iteration, final List<Integer> data) {
    if (iteration >= this.model.getNumOfHiddenLayers()) {
      return;
    }

    // receive responsible units for each hidden layer
    this.responsibleUnits.add(data);
  }

  @Override
  public final void run(final int iteration) {
    if (iteration == 0) {
      // generate random weights
      for (int i = 0; i < this.model.getNumOfWeights(); ++i) {
        randomizeWeights(this.model.getWeights(i), this.responsibleUnits.get(i));
        zeroWeights(this.model.getPrevWeights(i), this.responsibleUnits.get(i));
      }
    }

    if (this.feedForward) {
      // feed forward
      if (this.step == 0) {
        final Row row = this.rows.get(iteration / (this.model.getNumOfWeights() * 2));
        final List<Double> input = new ArrayList<>(row.getFeature().size() + 1);
        input.add(0.0);

        for (int i = 0; i < row.getFeature().size(); ++i) {
          input.add(row.getFeature().getQuick(i));
        }

        this.model.setUnits(0, input);

        final List<Double> output = new ArrayList<>(2);
        output.add(0.0);
        output.add(row.getOutput());
        this.model.setTarget(output);
      }

      final List<Double> inputUnits = this.model.getUnits(this.step);
      final List<Double> hiddenUnits = this.model.getUnits(this.step + 1);
      final List<List<Double>> inputWeights = this.model.getWeights(this.step);
      layerForward(inputUnits, hiddenUnits, inputWeights, this.responsibleUnits.get(this.step));
    } else {
      // back propagation
      if (step == 0) {
        // Compute error on output units.
        final List<Double> outputDelta = this.model.getDelta(this.model.getNumOfDelta() - 1);
        final List<Double> target = this.model.getTarget();
        final List<Double> outputUnits = this.model.getUnits(this.model.getNumOfLayers() - 1);
        final int nOutput = this.model.getNumOfOutput();
        final List<Double> err = new ArrayList<>(1);
        outputError(outputDelta, target, outputUnits, nOutput, err);

        // Adjust hidden weights.
        final List<Double> hiddenUnits = this.model.getUnits(this.model.getNumOfLayers() - 2);
        final int nHidden = this.model.getNumOfHiddenUnits().get(this.model.getNumOfHiddenLayers() - 1);
        final List<List<Double>> hiddenWeights = this.model.getWeights(this.model.getNumOfWeights() - 1);
        final List<List<Double>> hiddenPrevWeights = this.model.getPrevWeights(this.model.getNumOfWeights() - 1);
        adjustWeights(outputDelta, nOutput, hiddenUnits, nHidden, hiddenWeights, hiddenPrevWeights,
            this.eta, this.momentum);
      } else {
        // Compute error on hidden units.
        final List<Double> hiddenDelta = this.model.getDelta(this.model.getNumOfDelta() - 1 - this.step);
        final int nHidden = this.model.getNumOfHiddenUnits().get(this.model.getNumOfHiddenLayers() - this.step);
        final List<Double> outputDelta = this.model.getDelta(this.model.getNumOfDelta() - this.step);
        //final int nOutput = this.model.getNumOfHiddenUnits().get(this.model.getNumOfHiddenLayers() - this.step);
        final List<List<Double>> hiddenWeights = this.model.getWeights(this.model.getNumOfWeights() - this.step);
        final List<Double> hiddenUnits = this.model.getUnits(this.model.getNumOfLayers() - 1 - this.step);
        final List<Double> err = new ArrayList<>(1);
        hiddenError(hiddenDelta, nHidden, outputDelta, this.model.getNumOfOutput(), hiddenWeights, hiddenUnits, err);

        // Adjust input weights.
        final List<Double> inputUnits = this.model.getUnits(this.model.getNumOfLayers() - 2 - this.step);
        final int nInput = this.model.getNumOfInput();
        final List<List<Double>> inputWeights = this.model.getWeights(this.model.getNumOfWeights() - 1 - this.step);
        final List<List<Double>> inputPrevWeights
            = this.model.getPrevWeights(this.model.getNumOfWeights() - 1 - this.step);
        adjustWeights(hiddenDelta, nHidden, inputUnits, nInput, inputWeights, inputPrevWeights,
            this.eta, this.momentum);
      }
    }
  }

  @Override
  public List<Double> sendGatherData(final int iteration) {
    return null;
  }

  private void layerForward(final List<Double> l1, final List<Double> l2, final List<List<Double>> conn,
                            final List<Integer> responsibleUnits) {
    // Set up thresholding unit
    l1.set(0, 1.0);

    // For each unit in second layer
    for (final int i : responsibleUnits) {
      // Compute weighted sum of its inputs
      double sum = 0.0;

      for (int j = 0; j <= l1.size(); ++j) {
        sum += conn.get(j).get(i) * l1.get(j);
      }

      l2.set(i, squash(sum));
    }
  }

  private void outputError(final List<Double> delta, final List<Double> target, final List<Double> output,
                           final int nOutput, final List<Double> err) {
    double errSum = 0.0;

    for (int i = 1; i <= nOutput; i++) {
      final double o = output.get(i);
      final double t = target.get(i);
      delta.set(i, o * (1.0 - o) * (t - o));
      errSum += abs(delta.get(i));
    }

    err.add(errSum);
  }

  private void hiddenError(final List<Double> hiddenDelta, final int nHidden, final List<Double> outputDelta,
                           final int nOutput, final List<List<Double>> hiddenWeights, final List<Double> hiddenUnits,
                           final List<Double> err) {
    double errSum = 0.0;

    for (int i = 1; i <= nHidden; ++i) {
      final double h = hiddenUnits.get(i);
      double sum = 0.0;

      for (int j = 1; j <= nOutput; ++j) {
        sum += outputDelta.get(j) * hiddenWeights.get(i).get(j);
      }

      hiddenDelta.set(i, h * (1.0 - h) * sum);
      errSum += abs(hiddenDelta.get(i));
    }

    err.add(errSum);
  }

  private void adjustWeights(List<Double> delta, int nL2, List<Double> unitsL1, int nL1, List<List<Double>> weights,
                             List<List<Double>> prevWeights, double eta, double momentum) {
    unitsL1.set(0, 1.0);

    for (int i = 1; i <= nL2; ++i) {
      for (int j = 0; j <= nL1; ++j) {
        double newDiffWeight = eta * delta.get(i) * unitsL1.get(j) + momentum * prevWeights.get(j).get(i);
        weights.get(j).set(i, weights.get(j).get(i) + newDiffWeight);
        prevWeights.get(j).set(i, newDiffWeight);
      }
    }
  }

  // The squashing function.  Currently, it's a sigmoid.
  private double squash(double x) {
    return (1.0 / (1.0 + exp(-x)));
  }

  // Return random number between 0.0 and 1.0
  private double drnd() {
    return random.nextDouble() / Double.POSITIVE_INFINITY;
  }

  // Return random number between -1.0 and 1.0
  private double dpn1() {
    return ((drnd() * 2.0) - 1.0);
  }

  private void randomizeWeights(final List<List<Double>> weights, final List<Integer> responsibleUnits) {
    for (int i = 0; i <= weights.size(); ++i) {
      // index 0 is for thresholding unit
      weights.get(i).set(0, dpn1());

      for (final Integer j : responsibleUnits) {
        weights.get(i).set(j, dpn1());
      }
    }
  }

  private void zeroWeights(final List<List<Double>> weights, final List<Integer> responsibleUnits) {
    for (int i = 0; i <= weights.size(); ++i) {
      // index 0 is for thresholding unit
      weights.get(i).set(0, 0.0);

      for (final Integer j : responsibleUnits) {
        weights.get(i).set(j, 0.0);
      }
    }
  }
}