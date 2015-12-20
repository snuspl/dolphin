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
package edu.snu.dolphin.dnn.layerparam.provider;

import edu.snu.dolphin.dnn.layers.LayerParameter;

/**
 * Interface for parameter provider that gathers parameter gradients and provides updated parameters.
 */
public interface ParameterProvider {

  /**
   * Pushes parameter gradients for an input batch.
   * @param batchSize the size of an input batch
   * @param parameterGradients parameter gradients sums for an input batch
   */
  void push(final int batchSize, final LayerParameter[] parameterGradients);

  /**
   * @return the updated parameters of the whole network.
   */
  LayerParameter[] pull();
}
