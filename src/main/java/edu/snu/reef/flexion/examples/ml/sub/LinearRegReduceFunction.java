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
package edu.snu.reef.flexion.examples.ml.sub;

import com.microsoft.reef.io.network.group.operators.Reduce;
import edu.snu.reef.flexion.examples.ml.data.LinearRegSummary;

import javax.inject.Inject;

public class LinearRegReduceFunction implements Reduce.ReduceFunction<LinearRegSummary> {

  @Inject
  public LinearRegReduceFunction() {
  }

  @Override
  public final LinearRegSummary apply(Iterable<LinearRegSummary> summaryList) {
    LinearRegSummary reducedSummary = null;
    for (final LinearRegSummary summary : summaryList) {
      if (reducedSummary==null) {
        reducedSummary = summary;
      } else {
        reducedSummary.plus(summary);
      }
    }
    return reducedSummary;
  }
}