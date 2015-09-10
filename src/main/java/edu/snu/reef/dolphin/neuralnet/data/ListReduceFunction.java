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
package edu.snu.reef.dolphin.neuralnet.data;

import org.apache.reef.io.network.group.api.operators.Reduce;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;

/**
 * MPI Reduce function for a generic Java List.
 * Aggregates several lists into one big list.
 */
public final class ListReduceFunction implements Reduce.ReduceFunction<List> {

  @Inject
  private ListReduceFunction() {
  }

  @Override
  public List apply(final Iterable<List> elements) {
    final List retList = new LinkedList();
    for (final List element : elements) {
      retList.addAll(element);
    }
    return retList;
  }
}
