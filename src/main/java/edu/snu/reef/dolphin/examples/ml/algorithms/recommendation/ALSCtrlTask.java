package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.data.ALSSummary;
import edu.snu.reef.dolphin.examples.ml.key.ItemNum;
import edu.snu.reef.dolphin.examples.ml.key.UserNum;
import edu.snu.reef.dolphin.examples.ml.parameters.FeatureNum;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceReceiver;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.Map;
import java.util.logging.Logger;

public final class ALSCtrlTask extends UserControllerTask
    implements DataBroadcastSender<ALSSummary>, DataReduceReceiver<Map<Integer, Vector>> {

  private final static Logger LOG = Logger.getLogger(ALSCtrlTask.class.getName());

  private final int featureNum;
  private final int maxIter;
  private final int userNum;
  private final int itemNum;
  private Matrix broadcastMatrix;
  private Matrix prevMatrix;
  private ALSSummary.UserItem userItem;

  @Inject
  public ALSCtrlTask(@Parameter(FeatureNum.class) final int featureNum,
                     KeyValueStore keyValueStore,
                     @Parameter(MaxIterations.class) final int maxIter) {
    this.featureNum = featureNum;
    this.maxIter = maxIter;
    this.userNum = keyValueStore.get(UserNum.class);
    this.itemNum = keyValueStore.get(ItemNum.class);
  }

  @Override
  public void initialize() {
    broadcastMatrix = new DenseMatrix(featureNum, userNum).assign(0.1);
    userItem = ALSSummary.UserItem.USER;
  }

  @Override
  public void run(int iteration) {
    return;
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration > maxIter;
  }

  @Override
  public ALSSummary sendBroadcastData(int iteration) {
    return new ALSSummary(broadcastMatrix, userItem);
  }

  @Override
  public void receiveReduceData(int iteration, Map<Integer, Vector> map) {
    prevMatrix = broadcastMatrix;
    broadcastMatrix = convertMapToMatrix(map);
    userItem = userItem == ALSSummary.UserItem.USER ?
               ALSSummary.UserItem.ITEM :
               ALSSummary.UserItem.USER;
  }

  private final Matrix convertMapToMatrix(Map<Integer, Vector> map) {
    Matrix matrix = new DenseMatrix(featureNum,
        userItem == ALSSummary.UserItem.USER ? itemNum : userNum);
    for (final Integer index : map.keySet()) {
      matrix.assignColumn(index, map.get(index));
    }

    return matrix;
  }

  @Override
  public void cleanup() {
    System.out.println(prevMatrix.transpose().times(broadcastMatrix));
  }
}
