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
package edu.snu.dolphin.dnn.data;

import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.util.Pair;

import javax.inject.Inject;
import java.util.Iterator;

/**
 * Reduce function for pairs of an integer and an array of layer parameters.
 * Sums the integers and element-wise sums layer parameters.
 */
public final class IntAndLayerParameterArrayPairReduceFunction
    implements Reduce.ReduceFunction<Pair<Integer, LayerParameter[]>> {

  @Inject
  private IntAndLayerParameterArrayPairReduceFunction() {
  }

  @Override
  public Pair<Integer, LayerParameter[]> apply(
      final Iterable<Pair<Integer, LayerParameter[]>> intAndLayerParameterArrayPair) {

    final Iterator<Pair<Integer, LayerParameter[]>> iterator = intAndLayerParameterArrayPair.iterator();
    int intSum;
    final LayerParameter[] layerParameterSum;

    Pair<Integer, LayerParameter[]> element = getNonEmpty(iterator);
    if (element == null) {
      return new Pair<>(0, new LayerParameter[0]);
    }
    intSum = element.getFirst();
    layerParameterSum = element.getSecond();

    element = getNonEmpty(iterator);
    while (element != null) {
      intSum += element.getFirst();
      final LayerParameter[] layerParameters = element.getSecond();
      if (layerParameters.length != layerParameterSum.length) {
        throw new RuntimeException("The number of layer parameters is not consistent");
      }
      for (int i = 0; i < layerParameters.length; ++i) {
        layerParameterSum[i].getWeightParam().addi(layerParameters[i].getWeightParam());
        layerParameterSum[i].getBiasParam().addi(layerParameters[i].getBiasParam());
      }
      element = getNonEmpty(iterator);
    }

    return new Pair<>(intSum, layerParameterSum);
  }

  /**
   * @param iterator an iterator of pairs of an integer and an array of layer parameters.
   * @return a non empty pair or {@code null} if such pair does not exist.
   */
  private Pair<Integer, LayerParameter[]> getNonEmpty(final Iterator<Pair<Integer, LayerParameter[]>> iterator) {
    while (iterator.hasNext()) {
      final Pair<Integer, LayerParameter[]> element = iterator.next();
      if (element.getFirst() > 0) {
        return element;
      }
    }
    return null;
  }
}
