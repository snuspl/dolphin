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

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.StageInfo;
import edu.snu.reef.dolphin.core.UserJobInfo;
import edu.snu.reef.dolphin.examples.ml.data.SvdDataParser;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;

/**
 * Job info for svd.
 */
public final class SvdJobInfo implements UserJobInfo {

  @Inject
  public SvdJobInfo() {
  }

  @Override
  public List<StageInfo> getStageInfoList() {
    final List<StageInfo> stageInfoList = new LinkedList<>();

    // Load the input matrix A
    stageInfoList.add(LoadMatrixStageBuilder.build());
    // Compute eigen values and eigen vectors in ATA for the column vectors of V
    stageInfoList.add(EigenStageBuilder.build());
    // Compute column vectors of U and and result
    stageInfoList.add(PostEigenStageBuilder.build());

    return stageInfoList;
  }

  @Override
  public Class<? extends DataParser> getDataParser() {
    return SvdDataParser.class;
  }
}