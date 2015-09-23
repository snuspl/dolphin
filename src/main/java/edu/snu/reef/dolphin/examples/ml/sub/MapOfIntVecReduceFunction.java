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
