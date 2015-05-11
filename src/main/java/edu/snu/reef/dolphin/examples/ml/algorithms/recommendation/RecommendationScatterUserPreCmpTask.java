package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import edu.snu.reef.dolphin.examples.ml.key.UserRatings;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterReceiver;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.network.util.Pair;

import javax.inject.Inject;
import java.util.*;
import java.util.logging.Logger;

public final class RecommendationScatterUserPreCmpTask extends UserComputeTask
    implements DataScatterReceiver<List<List<Rating>>> {

  private final static Logger LOG = Logger.getLogger(RecommendationScatterUserPreCmpTask.class.getName());

  private final KeyValueStore keyValueStore;
  private List<List<Rating>> userRatingListList;

  @Inject
  public RecommendationScatterUserPreCmpTask(final KeyValueStore keyValueStore) {
    this.keyValueStore = keyValueStore;
  }

  @Override
  public void run(int iteration) {
    final Map<Integer, Pair<Collection<Integer>, Vector>> userIndexToRatingVectorMap = new HashMap<>();

    for (final List<Rating> userRatingList : userRatingListList) {
      if (userRatingList.size() == 0) {
        continue;
      }

      final int userIndex = userRatingList.get(0).getUserIndex();
      TreeMap<Integer, Rating> sortedMap = new TreeMap<>();
      for (final Rating rating : userRatingList) {
        sortedMap.put(rating.getItemIndex(), rating);
      }

      final Vector ratingVector = new DenseVector(sortedMap.size());
      int index = 0;
      for (final Integer itemIndex : sortedMap.keySet()) {
        ratingVector.set(index++, sortedMap.get(itemIndex).getRatingScore());
      }

      userIndexToRatingVectorMap.put(userIndex,
          new Pair<Collection<Integer>, Vector>(sortedMap.keySet(), ratingVector));
    }

    keyValueStore.put(UserRatings.class, userIndexToRatingVectorMap);
  }

  @Override
  public void receiveScatterData(int iteration, List<List<Rating>> userRatingListList) {
    this.userRatingListList = userRatingListList;
  }
}
