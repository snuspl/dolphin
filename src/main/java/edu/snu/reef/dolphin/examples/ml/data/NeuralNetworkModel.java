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

import java.util.ArrayList;
import java.util.List;

public final class NeuralNetworkModel {
  private final List<List<Double>> units;
  private final List<Double> target;
  private final List<List<Double>> delta; // storage for unit error
  private final List<List<List<Double>>> weights; // weights from layer i to layer i+1
  private final List<List<List<Double>>> preWeights; // previous change on layer i to layer i+1 wgt

  private final int nHiddenLayers;
  private final List<Integer> nUnits;

  public NeuralNetworkModel(final List<Integer> nUnits) {
    this.nHiddenLayers = nUnits.size() - 2;
    this.nUnits = nUnits;

    this.units = new ArrayList<>(this.nHiddenLayers + 2); // input layer + hidden layers + output layer
    this.units.add(new ArrayList<Double>(getNumOfInput() + 1));

    for (int i = 0; i < this.nHiddenLayers; ++i) {
      this.units.add(new ArrayList<Double>(nUnits.get(i + 1) + 1));
    }

    this.units.add(new ArrayList<Double>(getNumOfOutput() + 1));

    this.target = new ArrayList<>(getNumOfOutput() + 1);

    this.delta = new ArrayList<>(this.nHiddenLayers + 1); // hidden layers + output layer

    for (int i = 0; i < this.nHiddenLayers; ++i) {
      this.delta.add(new ArrayList<Double>(nUnits.get(i) + 1));
    }

    this.delta.add(new ArrayList<Double>(getNumOfOutput() + 1));

    this.weights = new ArrayList<>(this.nHiddenLayers + 1);
    this.weights.add(new ArrayList<List<Double>>(getNumOfInput() + 1));

    for (int i = 0; i < getNumOfInput() + 1; ++i) {
      this.weights.get(0).add(new ArrayList<Double>(getNumOfInput() + 1));
    }

    for (int i = 0; i < this.nHiddenLayers - 1; ++i) {
      this.weights.add(new ArrayList<List<Double>>(nUnits.get(i) + 1));

      for (int j = 0; j < nUnits.get(i) + 1; ++j) {
        this.weights.get(i + 1).add(new ArrayList<Double>(nUnits.get(i + 1) + 1));
      }
    }

    this.weights.add(new ArrayList<List<Double>>(nUnits.get(this.nHiddenLayers - 1) + 1));

    for (int i = 0; i < nUnits.get(this.nHiddenLayers - 1) + 1; ++i) {
      this.weights.get(this.nHiddenLayers).add(new ArrayList<Double>(getNumOfOutput() + 1));
    }

    this.preWeights = new ArrayList<>(this.nHiddenLayers + 1);
    this.preWeights.add(new ArrayList<List<Double>>(getNumOfInput() + 1));

    for (int i = 0; i < getNumOfInput() + 1; ++i) {
      this.preWeights.get(0).add(new ArrayList<Double>(getNumOfInput() + 1));
    }

    for (int i = 0; i < this.nHiddenLayers - 1; ++i) {
      this.preWeights.add(new ArrayList<List<Double>>(nUnits.get(i) + 1));

      for (int j = 0; j < nUnits.get(i) + 1; ++j) {
        this.preWeights.get(i + 1).add(new ArrayList<Double>(nUnits.get(i + 1) + 1));
      }
    }

    this.preWeights.add(new ArrayList<List<Double>>(nUnits.get(this.nHiddenLayers - 1) + 1));

    for (int i = 0; i < nUnits.get(this.nHiddenLayers - 1) + 1; ++i) {
      this.preWeights.get(this.nHiddenLayers).add(new ArrayList<Double>(getNumOfOutput() + 1));
    }
  }

  public List<Double> getUnits(final int layerIndex) {
    return this.units.get(layerIndex);
  }

  public void setUnits(final int layerIndex, final List<Double> units) {
    this.units.get(layerIndex).clear();
    this.units.get(layerIndex).addAll(units);
  }

  public List<Double> getTarget() {
    return this.target;
  }

  public void setTarget(final List<Double> target) {
    this.target.clear();
    this.target.addAll(target);
  }

  public List<Double> getDelta(final int layerIndex) {
    return this.delta.get(layerIndex);
  }

  public void setDelta(final int layerIndex, final List<Double> delta) {
    this.delta.get(layerIndex).clear();
    this.delta.get(layerIndex).addAll(delta);
  }

  public List<List<Double>> getWeights(final int layerIndex) {
    return this.weights.get(layerIndex);
  }

  public List<List<Double>> getPrevWeights(final int layerIndex) {
    return this.preWeights.get(layerIndex);
  }

  public List<Integer> getNumOfUnits() {
    return this.nUnits;
  }

  public int getNumOfInput() {
    return this.nUnits.get(0);
  }

  public int getNumOfOutput() {
    return this.nUnits.get(this.nUnits.size() - 1);
  }

  public int getNumOfHiddenLayers() {
    return this.nHiddenLayers;
  }

  public List<Integer> getNumOfHiddenUnits() {
    return this.nUnits.subList(1, this.nHiddenLayers);
  }

  public int getNumOfLayers() {
    return this.nHiddenLayers + 2;
  }

  public int getNumOfDelta() {
    return this.delta.size();
  }

  public int getNumOfWeights() {
    return this.weights.size();
  }
}
