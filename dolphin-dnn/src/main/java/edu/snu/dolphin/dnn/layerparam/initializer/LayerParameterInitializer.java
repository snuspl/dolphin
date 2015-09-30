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
package edu.snu.dolphin.dnn.layerparam.initializer;

import edu.snu.dolphin.dnn.layers.LayerParameter;

/**
 * Interface for parameter initializer.
 *
 * The parameter initializer generates the initial parameter of the layer by the layer configuration.
 */
public interface LayerParameterInitializer {

  /**
   * @return the initial parameter of the layer.
   */
  LayerParameter generateInitialParameter();

  /**
   * @return the index of the layer.
   */
  int getIndex();
}