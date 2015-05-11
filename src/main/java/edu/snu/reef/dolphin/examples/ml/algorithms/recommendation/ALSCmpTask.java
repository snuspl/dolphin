package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.examples.ml.data.ALSSummary;
import edu.snu.reef.dolphin.examples.ml.key.ItemRatings;
import edu.snu.reef.dolphin.examples.ml.key.UserRatings;
import edu.snu.reef.dolphin.examples.ml.parameters.ALSLambda;
import edu.snu.reef.dolphin.examples.ml.parameters.FeatureNum;
import edu.snu.reef.dolphin.examples.ml.utils.MatrixUtils;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceSender;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.solver.ConjugateGradientSolver;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

public final class ALSCmpTask extends UserComputeTask
    implements DataBroadcastReceiver<ALSSummary>, DataReduceSender<Map<Integer, Vector>> {

  private final static Logger LOG = Logger.getLogger(ALSCmpTask.class.getName());

  private final int featureNum;
  private final KeyValueStore keyValueStore;
  private final double lambda;
  private final ConjugateGradientSolver solver;
  private Map<Integer, Vector> indexToVectorMap;
  private Matrix givenMatrix;
  private ALSSummary.UserItem userItem;


  @Inject
  public ALSCmpTask(@Parameter(FeatureNum.class) final int featureNum,
                    final KeyValueStore keyValueStore,
                    @Parameter(ALSLambda.class) final double lambda) {
    this.featureNum = featureNum;
    this.keyValueStore = keyValueStore;
    this.lambda = lambda;
    this.solver = new ConjugateGradientSolver();
  }

  @Override
  public void run(int iteration) {
    indexToVectorMap = (userItem == ALSSummary.UserItem.USER) ?
                       computeItemVector() :
                       computeUserVector();
  }

  private final Map<Integer, Vector> computeUserVector() {
    final Map<Integer, Vector> userIndexToVectorMap = new HashMap<>();
    final Map<Integer, Pair<Collection<Integer>, Vector>> userIndexToRatingVectorMap =
        keyValueStore.get(UserRatings.class);

    for (final Integer userIndex : userIndexToRatingVectorMap.keySet()) {
      final Collection<Integer> itemIndices = userIndexToRatingVectorMap.get(userIndex).first;
      final Vector ratingVector = userIndexToRatingVectorMap.get(userIndex).second;
      final Matrix filteredMatrix = MatrixUtils.viewColumns(givenMatrix, itemIndices);
      final int numberOfIndices = itemIndices.size();

      final SparseMatrix diagonalMatrix = new SparseMatrix(featureNum, featureNum);
      for (int index = 0; index < featureNum; index++) {
        diagonalMatrix.set(index, index, lambda * numberOfIndices);
      }
      final Matrix coefficientMatrix = filteredMatrix.times(filteredMatrix.transpose())
                                                     .plus(diagonalMatrix);
      final Vector constantVector = filteredMatrix.times(ratingVector);

      userIndexToVectorMap.put(userIndex, solver.solve(coefficientMatrix, constantVector));
    }

    return userIndexToVectorMap;
  }

  private final Map<Integer, Vector> computeItemVector() {
    final Map<Integer, Vector> itemIndexToVectorMap = new HashMap<>();
    final Map<Integer, Pair<Collection<Integer>, Vector>> itemIndexToRatingVectorMap =
        keyValueStore.get(ItemRatings.class);

    for (final Integer itemIndex : itemIndexToRatingVectorMap.keySet()) {
      final Collection<Integer> userIndices = itemIndexToRatingVectorMap.get(itemIndex).first;
      final Vector ratingVector = itemIndexToRatingVectorMap.get(itemIndex).second;
      final Matrix filteredMatrix = MatrixUtils.viewColumns(givenMatrix, userIndices);
      final int numberOfIndices = userIndices.size();

      final SparseMatrix diagonalMatrix = new SparseMatrix(featureNum, featureNum);
      for (int index = 0; index < featureNum; index++) {
        diagonalMatrix.set(index, index, lambda * numberOfIndices);
      }
      final Matrix coefficientMatrix = filteredMatrix.times(filteredMatrix.transpose())
                                                     .plus(diagonalMatrix);
      final Vector constantVector = filteredMatrix.times(ratingVector);

      itemIndexToVectorMap.put(itemIndex, solver.solve(coefficientMatrix, constantVector));
    }

    return itemIndexToVectorMap;
  }

  @Override
  public void receiveBroadcastData(int iteration, ALSSummary alsSummary) {
    userItem = alsSummary.getUserItem();
    givenMatrix = alsSummary.getMatrix();
  }

  @Override
  public Map<Integer, Vector> sendReduceData(int iteration) {
    return indexToVectorMap;
  }
}
