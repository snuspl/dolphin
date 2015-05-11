package edu.snu.reef.dolphin.examples.ml.sub;

import com.microsoft.reef.io.network.group.operators.Reduce;
import org.apache.mahout.math.Vector;

import javax.inject.Inject;
import java.util.Map;

public final class MapOfIntVecReduceFunction implements Reduce.ReduceFunction<Map<Integer, Vector>> {

  @Inject
  public MapOfIntVecReduceFunction() {
  }

  @Override
  public final Map<Integer, Vector> apply(final Iterable<Map<Integer, Vector>> mapList) {
    Map<Integer, Vector> retMap = null;

    for (final Map<Integer, Vector> map : mapList) {
      if (retMap == null) {
        retMap = map;
        continue;
      }

      for (final int index : map.keySet()) {
        retMap.put(index, map.get(index));
      }
    }

    return retMap;
  }
}
