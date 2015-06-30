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
package edu.snu.reef.dolphin.examples.simple;

import edu.snu.reef.dolphin.core.*;
import org.apache.reef.io.serialization.SerializableCodec;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;

/**
 * JobInfo class for the simple example application.
 */
public final class SimpleJobInfo implements UserJobInfo{

  @Inject
  private SimpleJobInfo(){
  }

  @Override
  public List<StageInfo> getStageInfoList() {
    final List<StageInfo> stageInfoList = new LinkedList<>();
    stageInfoList.add(
        StageInfo.newBuilder(SimpleCmpTask.class, SimpleCtrlTask.class, SimpleCommGroup.class)
            .setBroadcast(SerializableCodec.class)
            .setReduce(SerializableCodec.class, SimpleReduceFunction.class)
            .build());
    return stageInfoList;
  }

  @Override
  public Class<? extends DataParser> getDataParser() {
    return SimpleDataParser.class;
  }

  /**
   * Name for a communication group used by the job.
   */
  @NamedParameter(doc = "Name for a communication group used by the job.")
  private static final class SimpleCommGroup implements Name<String> {
  }
}
