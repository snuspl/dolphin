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
