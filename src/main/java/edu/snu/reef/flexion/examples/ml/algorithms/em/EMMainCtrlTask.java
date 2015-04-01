package edu.snu.reef.flexion.examples.ml.algorithms.em;

import edu.snu.reef.flexion.core.KeyValueStore;
import edu.snu.reef.flexion.core.UserControllerTask;
import edu.snu.reef.flexion.examples.ml.converge.ConvergenceCondition;
import edu.snu.reef.flexion.examples.ml.data.Centroid;
import edu.snu.reef.flexion.examples.ml.data.ClusterStats;
import edu.snu.reef.flexion.examples.ml.data.ClusterSummary;
import edu.snu.reef.flexion.examples.ml.data.Covariance;
import edu.snu.reef.flexion.examples.ml.key.Centroids;
import edu.snu.reef.flexion.examples.ml.parameters.IsCovarianceShared;
import edu.snu.reef.flexion.examples.ml.parameters.MaxIterations;
import edu.snu.reef.flexion.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.flexion.groupcomm.interfaces.DataReduceReceiver;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class EMMainCtrlTask extends UserControllerTask
        implements DataReduceReceiver<Map<Integer, ClusterStats>>, DataBroadcastSender<List<ClusterSummary>> {

    private static final Logger LOG = Logger.getLogger(EMMainCtrlTask.class.getName());

    /**
     * Check function to determine whether algorithm has converged or not.
     * This is separate from the default stop condition,
     * which is based on the number of iterations made.
     */
    private final ConvergenceCondition convergenceCondition;

    /**
     * Maximum number of iterations allowed before job stops
     */
    private final int maxIterations;

    /**
     * Aggregated statistics of each cluster received from Compute Task
     */
    private Map<Integer, ClusterStats> clusterStatsMap = new HashMap<>();

    /**
     * List of the centroids of the clusters passed from the preprocess stage
     * Will be updated for each iteration
     */
    private List<Centroid> centroids = new ArrayList<Centroid>();

    /**
     * List of the summaries of the clusters to distribute to Compute Tasks
     * Will be updated for each iteration
     */
    private List<ClusterSummary> clusterSummaries = new ArrayList<ClusterSummary>();

    /**
     * Whether to share a covariance matrix among clusters or not
     */
    private final boolean isCovarianceShared;

    /**
     * This class is instantiated by TANG
     *
     * Constructs the Controller Task for EM
     *
     * @param convergenceCondition  conditions for checking convergence of algorithm
     * @param maxIterations maximum number of iterations allowed before job stops
     * @param isCovarianceShared    whether clusters share one covariance matrix or not
     */
    @Inject
    public EMMainCtrlTask(final ConvergenceCondition convergenceCondition,
                          @Parameter(MaxIterations.class) final int maxIterations,
                          @Parameter(IsCovarianceShared.class) final boolean isCovarianceShared) {

        this.convergenceCondition = convergenceCondition;
        this.maxIterations = maxIterations;
        this.isCovarianceShared = isCovarianceShared;
    }

    /**
     * Receive initial centroids from the preprocess task
     * @param keyValueStore
     */
    public void initialize(KeyValueStore keyValueStore) {

        // Load the initial centroids from the previous stage
        centroids = keyValueStore.get(Centroids.class);

        // Initialize cluster summaries
        int numClusters = centroids.size();
        for(int i=0; i<numClusters; i++){
            Centroid centroid = centroids.get(i);
            int clusterID = centroid.getClusterId();
            int dimension = centroid.dimension();
            clusterSummaries.add(new ClusterSummary(clusterID, 1.0, centroid,
                    new Covariance(clusterID, DiagonalMatrix.identity(dimension))));
        }

    }

    @Override
    public void run(int iteration) {

        // Compute the shared covariance matrix if necessary
        Matrix covarianceMatrix = null;
        if (isCovarianceShared) {
            ClusterStats clusterStatsSum = null;
            for (final Integer id : clusterStatsMap.keySet()) {
                final ClusterStats clusterStats = clusterStatsMap.get(id);
                if (clusterStatsSum==null) {
                    clusterStatsSum = new ClusterStats(clusterStats, true);
                } else {
                    clusterStatsSum.add(clusterStats);
                }
            }
            if (clusterStatsSum!=null) {
                covarianceMatrix = clusterStatsSum.computeCovariance();
            }
        }

        // Compute new prior probability, centroids, and covariance matrices
        for (final Integer clusterID : clusterStatsMap.keySet()) {
            final ClusterStats clusterStats = clusterStatsMap.get(clusterID);
            final Centroid newCentroid = new Centroid(clusterID, clusterStats.computeMean());
            Covariance newCovariance = null;
            if (isCovarianceShared) {
                newCovariance = new Covariance(clusterID, covarianceMatrix);
            } else {
                newCovariance = new Covariance(clusterID, clusterStats.computeCovariance());
            }
            final double newPrior = clusterStats.probSum; //unnormalized prior

            centroids.set(clusterID, newCentroid);
            clusterSummaries.set(clusterID, new ClusterSummary(clusterID, newPrior, newCentroid, newCovariance));
        }

        LOG.log(Level.INFO, "********* Centroids after {0} iterations*********", iteration + 1);
        LOG.log(Level.INFO, "" + clusterSummaries);
        LOG.log(Level.INFO, "********* Centroids after {0} iterations*********", iteration + 1);

    }

    @Override
    public boolean isTerminated(int iteration) {
        return convergenceCondition.checkConvergence(centroids)
            || (iteration > maxIterations); // First two iterations are used for initialization

    }

    @Override
    public void receiveReduceData(Map<Integer, ClusterStats> data) {
        clusterStatsMap = data;
    }

    @Override
    public List<ClusterSummary> sendBroadcastData(int iteration) {
        return clusterSummaries;
    }
}
