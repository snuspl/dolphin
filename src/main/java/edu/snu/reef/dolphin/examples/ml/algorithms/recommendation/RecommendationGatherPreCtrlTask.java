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
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import edu.snu.reef.dolphin.examples.ml.key.ItemNum;
import edu.snu.reef.dolphin.examples.ml.key.Ratings;
import edu.snu.reef.dolphin.examples.ml.key.UserNum;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

public final class RecommendationGatherPreCtrlTask extends UserControllerTask
    implements DataGatherReceiver<List<Rating>> {

  private static final Logger LOG = Logger.getLogger(RecommendationGatherPreCtrlTask.class.getName());

  private final KeyValueStore keyValueStore;
  private final List<Rating> ratings;

  @Inject
  public RecommendationGatherPreCtrlTask(final KeyValueStore keyValueStore) {
    this.keyValueStore = keyValueStore;
    this.ratings = new LinkedList<>();
  }

  @Override
  public void run(int iteration) {
    return;
  }

  @Override
  public void cleanup() {
    keyValueStore.put(Ratings.class, ratings);
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration > 0;
  }

  @Override
  public void receiveGatherData(int iteration, List<List<Rating>> ratingListList) {
    int userIndexMax = -1;
    int itemIndexMax = -1;
    for (final List<Rating> ratingList : ratingListList) {
      for (final Rating rating : ratingList) {
        ratings.add(rating);
        userIndexMax = Math.max(userIndexMax, rating.getUserIndex());
        itemIndexMax = Math.max(itemIndexMax, rating.getItemIndex());
      }
    }

    keyValueStore.put(UserNum.class, userIndexMax + 1);
    keyValueStore.put(ItemNum.class, itemIndexMax + 1);
  }
}
