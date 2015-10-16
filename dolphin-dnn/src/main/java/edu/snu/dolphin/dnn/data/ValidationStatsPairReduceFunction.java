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
package edu.snu.dolphin.dnn.data;

import edu.snu.dolphin.dnn.util.ValidationStats;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.util.Pair;

import javax.inject.Inject;

/**
 * Reduce function for training/cross validation statistics.
 * Simply adds the number of processed samples and the number of correct samples.
 */
public final class ValidationStatsPairReduceFunction
    implements Reduce.ReduceFunction<Pair<ValidationStats, ValidationStats>> {

  @Inject
  private ValidationStatsPairReduceFunction() {
  }

  @Override
  public Pair<ValidationStats, ValidationStats> apply(
      final Iterable<Pair<ValidationStats, ValidationStats>> validationStatsPairIterable) {
    int firstTotalNum = 0;
    int firstCorrectNum = 0;
    int secondTotalNum = 0;
    int secondCorrectNum = 0;

    for (final Pair<ValidationStats, ValidationStats> validationStatsPair : validationStatsPairIterable) {
      firstTotalNum += validationStatsPair.getFirst().getTotalNum();
      firstCorrectNum += validationStatsPair.getFirst().getCorrectNum();
      secondTotalNum += validationStatsPair.getSecond().getTotalNum();
      secondCorrectNum += validationStatsPair.getSecond().getCorrectNum();
    }

    return new Pair<>(new ValidationStats(firstTotalNum, firstCorrectNum),
        new ValidationStats(secondTotalNum, secondCorrectNum));
  }
}
