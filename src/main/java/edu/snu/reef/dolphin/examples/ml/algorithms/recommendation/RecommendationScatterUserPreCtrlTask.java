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
import edu.snu.reef.dolphin.examples.ml.key.Ratings;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterSender;

import javax.inject.Inject;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public final class RecommendationScatterUserPreCtrlTask extends UserControllerTask
    implements DataScatterSender<List<Rating>> {

  private final static Logger LOG = Logger.getLogger(RecommendationScatterUserPreCtrlTask.class.getName());

  private final KeyValueStore keyValueStore;
  private final Map<Integer, List<Rating>> userCentricRatings;

  @Inject
  public RecommendationScatterUserPreCtrlTask(final KeyValueStore keyValueStore) {
    this.keyValueStore = keyValueStore;
    this.userCentricRatings = new HashMap<>();
  }

  @Override
  public void initialize() {
    final List<Rating> ratings = keyValueStore.get(Ratings.class);
    for (final Rating rating : ratings) {
      if (!userCentricRatings.containsKey(rating.getUserIndex())) {
        userCentricRatings.put(rating.getUserIndex(), new LinkedList<Rating>());
      }
      userCentricRatings.get(rating.getUserIndex()).add(rating);
    }
  }

  @Override
  public void run(int iteration) {
    return;
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration > 0;
  }


  @Override
  public List<List<Rating>> sendScatterData(int iteration) {
    List<List<Rating>> ratingListList = new LinkedList<>();
    for (final List<Rating> ratingList : userCentricRatings.values()) {
      ratingListList.add(ratingList);
    }
    return ratingListList;
  }
}
