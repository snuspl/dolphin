package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.ParseException;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherSender;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Logger;

public final class RecommendationGatherPreCmpTask extends UserComputeTask
    implements DataGatherSender<List<Rating>> {
  private final static Logger LOG = Logger.getLogger(RecommendationGatherPreCmpTask.class.getName());

  private List<Rating> ratings;
  private final DataParser<List<Rating>> dataParser;

  @Inject
  public RecommendationGatherPreCmpTask(final DataParser<List<Rating>> dataParser) {
    this.dataParser = dataParser;
  }

  @Override
  public void initialize() throws ParseException {
    ratings = dataParser.get();
  }

  @Override
  public void run(int iteration) {
    // do nothing
  }

  @Override
  public List<Rating> sendGatherData(int iteration) {
    return ratings;
  }
}
