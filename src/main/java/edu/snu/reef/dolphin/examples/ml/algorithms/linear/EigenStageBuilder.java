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
package edu.snu.reef.dolphin.examples.ml.algorithms.linear;

import edu.snu.reef.dolphin.core.StageInfo;
import edu.snu.reef.dolphin.examples.ml.sub.VectorCodec;
import org.apache.reef.io.serialization.SerializableCodec;

/**
 * Stage builder for computing eigen vectors and values used in svd.
 */
public final class EigenStageBuilder {
  private EigenStageBuilder() {
  }

  public static StageInfo build() {
    return StageInfo.newBuilder(EigenCmpTask.class, EigenCtrlTask.class, EigenCommGroup.class)
        .setScatter(SerializableCodec.class)
        .setBroadcast(VectorCodec.class)
        .setGather(SerializableCodec.class).build();
  }
}