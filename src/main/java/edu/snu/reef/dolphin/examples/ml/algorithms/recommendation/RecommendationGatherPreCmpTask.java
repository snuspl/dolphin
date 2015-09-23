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
