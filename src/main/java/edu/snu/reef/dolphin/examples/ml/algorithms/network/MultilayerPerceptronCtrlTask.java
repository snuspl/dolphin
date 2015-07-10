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

import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.data.NeuralNetworkModel;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterSender;

import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultilayerPerceptronCtrlTask extends UserControllerTask
    implements DataBroadcastSender<List<Double>>,
    DataScatterSender<Integer>,
    DataGatherReceiver<List<Double>> {
  private final NeuralNetworkModel model;
  private final List<Double> data;

  @Inject
  public MultilayerPerceptronCtrlTask() {
    // hidden layer
    final int nHidden = 5;
    final List<Integer> nUnits = new ArrayList<>(nHidden + 2);
    final Random random = new Random();
    nUnits.add(3);

    for (int i = 0; i < nHidden; ++i) {
      nUnits.add(random.nextInt());
    }

    nUnits.add(1);
    this.model = new NeuralNetworkModel(nHidden, nUnits);
    this.data = new ArrayList<>();
  }

  @Override
  public final void run(final int iteration) {
  }

  @Override
  public final boolean isTerminated(final int iteration) {
    return iteration > 1000;
  }

  @Override
  public final List<Double> sendBroadcastData(final int iteration) {
    if (iteration == 0) {
      final List<Integer> nUnits = this.model.getNumOfUnits();
      final List<Double> ret = new ArrayList<>(nUnits.size());

      for (final Integer i : nUnits) {
        ret.add(i.doubleValue());
      }

      return ret;
    } else {
      return this.data;
    }
  }

  @Override
  public List<Integer> sendScatterData(final int iteration) {
    if (iteration >= this.model.getNumOfLayers()) {
      return Collections.EMPTY_LIST;
    }

    final int nUnits = this.model.getNumOfUnits().get(iteration);
    final List<Integer> ret = new ArrayList<>(nUnits);

    // index 0 is for thresholding unit
    for (int i = 1; i <= nUnits; ++i) {
      ret.add(i);
    }

    return ret;
  }

  @Override
  public void receiveGatherData(final int iteration, final List<List<Double>> data) {
    this.data.clear();

    for (final List<Double> i : data) {
      this.data.addAll(i);
    }
  }
}
