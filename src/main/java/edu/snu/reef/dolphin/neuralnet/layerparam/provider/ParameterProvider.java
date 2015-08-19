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
package edu.snu.reef.dolphin.neuralnet.layerparam.provider;

import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Interface for parameter provider that gathers activation values and gradients and provides updated parameters.
 */
public interface ParameterProvider {

  /**
   * Pushes activation values and gradients for each training input.
   * @param activations activation values of the training input.
   * @param gradients error gradient vectors of the training input.
   */
  void push(final List<INDArray> activations, final List<INDArray> gradients);

  /**
   * Returns the updated parameters of the whole network.
   * @return the updated parameters.
   */
  LayerParameter[] pull();
}
