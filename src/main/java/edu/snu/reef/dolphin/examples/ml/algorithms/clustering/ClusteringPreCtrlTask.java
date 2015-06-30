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
package edu.snu.reef.dolphin.examples.ml.algorithms.clustering;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.key.Centroids;
import edu.snu.reef.dolphin.examples.ml.parameters.NumberOfClusters;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherReceiver;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;

/**
 * User-defined compute task class for the preprocessing stage of the K-means algorithm.
 * The controller task sample initial centroids of clusters from samples received from compute tasks.
 */
public final class ClusteringPreCtrlTask extends UserControllerTask
    implements DataGatherReceiver<List<Vector>> {

  /**
   * Number of clusters learned by the clustering algorithm.
   */
  private final int numberOfClusters;

  /**
   * Initial centroids passed from Compute Tasks.
   */
  private List<Vector> initialCentroids = null;
  private final KeyValueStore keyValueStore;

  /**
   * This class is instantiated by TANG.
   *
   * @param keyValueStore Key-value store object.
   * @param numberOfClusters Number of clusters learned by the clustering algorithm.
   */
  @Inject
  private ClusteringPreCtrlTask(
      final KeyValueStore keyValueStore,
      @Parameter(NumberOfClusters.class) final int numberOfClusters) {
    this.keyValueStore = keyValueStore;
    this.numberOfClusters = numberOfClusters;
  }

  @Override
  public void run(int iteration) {
    //do nothing
  }

  @Override
  public void cleanup() {

    // pass initial centroids to the main process
    keyValueStore.put(Centroids.class, initialCentroids);
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration > 0;

  }

  @Override
  public void receiveGatherData(int iteration, List<List<Vector>> initialCentroids) {
    final List<Vector> points = new LinkedList<>();

    // Flatten the given list of lists
    for(List<Vector> list : initialCentroids) {
      for(Vector vector: list){
        points.add(vector);
      }
    }

    //sample initial centroids
    this.initialCentroids = ClusteringPreCmpTask.sample(points, numberOfClusters);
  }
}
