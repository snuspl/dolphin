package edu.snu.reef.flexion.examples.ml.algorithms.kmeans;


import edu.snu.reef.flexion.core.DataParser;
import edu.snu.reef.flexion.core.UserJobInfo;
import edu.snu.reef.flexion.core.StageInfo;
import edu.snu.reef.flexion.examples.ml.algorithms.ClusteringPreCtrlTask;
import edu.snu.reef.flexion.examples.ml.algorithms.ClusteringDataParser;
import edu.snu.reef.flexion.examples.ml.algorithms.ClusteringPreCmpTask;
import edu.snu.reef.flexion.examples.ml.algorithms.ClusteringPreCommGroup;
import edu.snu.reef.flexion.examples.ml.sub.CentroidListCodec;
import edu.snu.reef.flexion.examples.ml.sub.MapOfIntVSumCodec;
import edu.snu.reef.flexion.examples.ml.sub.MapOfIntVSumReduceFunction;
import edu.snu.reef.flexion.examples.ml.sub.VectorListCodec;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;

public final class KMeansJobInfo implements UserJobInfo {

    @Inject
    public KMeansJobInfo(){
    }

    @Override
    public List<StageInfo> getStageInfoList() {

        List<StageInfo> stageInfoList = new LinkedList<>();

        // preprocess: initialize the centroids of clusters
        stageInfoList.add(new StageInfo(ClusteringPreCmpTask.class, ClusteringPreCtrlTask.class, ClusteringPreCommGroup.class)
                .setGather(VectorListCodec.class));

        // main process: adjust the centroids of clusters
        stageInfoList.add(new StageInfo(KMeansMainCmpTask.class, KMeansMainCtrlTask.class, KMeansMainCommGroup.class)
                .setBroadcast(CentroidListCodec.class)
                .setReduce(MapOfIntVSumCodec.class, MapOfIntVSumReduceFunction.class));

        return stageInfoList;
    }

    @Override
    public Class<? extends DataParser> getDataParser() {
        return ClusteringDataParser.class;
    }
}
