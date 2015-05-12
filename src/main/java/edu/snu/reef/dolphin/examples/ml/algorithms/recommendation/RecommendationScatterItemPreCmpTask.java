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
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import edu.snu.reef.dolphin.examples.ml.key.ItemRatings;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterReceiver;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.network.util.Pair;

import javax.inject.Inject;
import java.util.*;
import java.util.logging.Logger;

public final class RecommendationScatterItemPreCmpTask extends UserComputeTask
    implements DataScatterReceiver<List<List<Rating>>> {

  private final static Logger LOG = Logger.getLogger(RecommendationScatterItemPreCmpTask.class.getName());

  private final KeyValueStore keyValueStore;
  private List<List<Rating>> itemRatingListList;

  @Inject
  public RecommendationScatterItemPreCmpTask(final KeyValueStore keyValueStore) {
    this.keyValueStore = keyValueStore;
  }

  @Override
  public void run(int iteration) {
    final Map<Integer, Pair<Collection<Integer>, Vector>> itemIndexToRatingVectorMap = new HashMap<>();

    for (final List<Rating> itemRatingList : itemRatingListList) {
      if (itemRatingList.size() == 0) {
        continue;
      }

      final int itemIndex = itemRatingList.get(0).getItemIndex();
      TreeMap<Integer, Rating> sortedMap = new TreeMap<>();
      for (final Rating rating : itemRatingList) {
        sortedMap.put(rating.getUserIndex(), rating);
      }

      final Vector ratingVector = new DenseVector(sortedMap.size());
      int index = 0;
      for (final Integer userIndex : sortedMap.keySet()) {
        ratingVector.set(index++, sortedMap.get(userIndex).getRatingScore());
      }

      itemIndexToRatingVectorMap.put(itemIndex,
          new Pair<Collection<Integer>, Vector>(sortedMap.keySet(), ratingVector));
    }

    keyValueStore.put(ItemRatings.class, itemIndexToRatingVectorMap);
  }

  @Override
  public void receiveScatterData(int iteration, List<List<Rating>> itemRatingListList) {
    this.itemRatingListList = itemRatingListList;
  }
}
