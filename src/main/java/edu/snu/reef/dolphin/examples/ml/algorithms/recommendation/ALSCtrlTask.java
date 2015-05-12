/**
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
package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.data.ALSSummary;
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import edu.snu.reef.dolphin.examples.ml.key.ItemNum;
import edu.snu.reef.dolphin.examples.ml.key.Ratings;
import edu.snu.reef.dolphin.examples.ml.key.UserNum;
import edu.snu.reef.dolphin.examples.ml.parameters.FeatureNum;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceReceiver;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class ALSCtrlTask extends UserControllerTask
    implements DataBroadcastSender<ALSSummary>, DataReduceReceiver<Map<Integer, Vector>> {

  private final static Logger LOG = Logger.getLogger(ALSCtrlTask.class.getName());

  private final int featureNum;
  private final int maxIter;
  private final int userNum;
  private final int itemNum;
  private final KeyValueStore keyValueStore;
  private final Random random;
  private Matrix broadcastMatrix;
  private Matrix prevMatrix;
  private ALSSummary.UserItem userItem;

  @Inject
  public ALSCtrlTask(@Parameter(FeatureNum.class) final int featureNum,
                     KeyValueStore keyValueStore,
                     @Parameter(MaxIterations.class) final int maxIter) {
    this.featureNum = featureNum;
    this.maxIter = maxIter;
    this.userNum = keyValueStore.get(UserNum.class);
    this.itemNum = keyValueStore.get(ItemNum.class);
    this.keyValueStore = keyValueStore;
    this.random = new Random();
  }

  @Override
  public void initialize() {
    final List<Rating> ratings = keyValueStore.get(Ratings.class);
    final Map<Integer, Double> indexToSumMap = new HashMap<>();
    for (final Rating rating : ratings) {
      if (!indexToSumMap.containsKey(rating.getItemIndex())) {
        indexToSumMap.put(rating.getItemIndex(), 0D);
      }

      indexToSumMap.put(rating.getItemIndex(),
                        indexToSumMap.get(rating.getItemIndex()) + rating.getRatingScore());
    }

    broadcastMatrix = new DenseMatrix(featureNum, itemNum);
    for (int itemIndex = 0; itemIndex < itemNum; itemIndex++) {
      for (int featureIndex = 0; featureIndex < featureNum; featureIndex++) {
        if (indexToSumMap.containsKey(itemIndex)) {
          broadcastMatrix.set(featureIndex, itemIndex, indexToSumMap.get(itemIndex) + random.nextDouble());
        } else {
          broadcastMatrix.set(featureIndex, itemIndex, random.nextDouble());
        }
      }
    }
    userItem = ALSSummary.UserItem.ITEM;
  }

  @Override
  public void run(int iteration) {
    return;
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration > maxIter;
  }

  @Override
  public ALSSummary sendBroadcastData(int iteration) {
    return new ALSSummary(broadcastMatrix, userItem);
  }

  @Override
  public void receiveReduceData(int iteration, Map<Integer, Vector> map) {
    prevMatrix = broadcastMatrix;
    broadcastMatrix = convertMapToMatrix(map);
    userItem = userItem == ALSSummary.UserItem.USER ?
               ALSSummary.UserItem.ITEM :
               ALSSummary.UserItem.USER;
  }

  private final Matrix convertMapToMatrix(Map<Integer, Vector> map) {
    Matrix matrix = new DenseMatrix(featureNum,
        userItem == ALSSummary.UserItem.USER ? itemNum : userNum);
    for (final Integer index : map.keySet()) {
      matrix.assignColumn(index, map.get(index));
    }

    return matrix;
  }

  @Override
  public void cleanup() {
    if (userItem == ALSSummary.UserItem.USER) {
      LOG.log(Level.INFO, "Final User Matrix: {0}", broadcastMatrix);
      LOG.log(Level.INFO, "Final Item Matrix: {0}", prevMatrix);
    } else {
      LOG.log(Level.INFO, "Final User Matrix: {0}", prevMatrix);
      LOG.log(Level.INFO, "Final Item Matrix: {0}", broadcastMatrix);
    }
  }
}
