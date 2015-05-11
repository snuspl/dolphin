package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.KeyValueStore;
import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.examples.ml.data.ALSSummary;
import edu.snu.reef.dolphin.examples.ml.key.UserNum;
import edu.snu.reef.dolphin.examples.ml.parameters.FeatureNum;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.logging.Logger;

public final class ALSCtrlTask extends UserControllerTask
    implements DataBroadcastSender<ALSSummary> {

  private final static Logger LOG = Logger.getLogger(ALSCtrlTask.class.getName());

  private final int featureNum;
  private final int maxIter;
  private final int userNum;

  @Inject
  public ALSCtrlTask(@Parameter(FeatureNum.class) final int featureNum,
                     KeyValueStore keyValueStore,
                     @Parameter(MaxIterations.class) final int maxIter) {
    this.featureNum = featureNum;
    this.maxIter = maxIter;
    this.userNum = keyValueStore.get(UserNum.class);
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
  public ALSSummary sendBroadcastData(int iteration) {
    Matrix matrix = new DenseMatrix(featureNum, userNum);
    return new ALSSummary(matrix.assign(0.1), ALSSummary.UserItem.USER);
  }
}
