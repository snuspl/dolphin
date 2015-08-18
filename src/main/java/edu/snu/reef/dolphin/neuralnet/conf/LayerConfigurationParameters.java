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
package edu.snu.reef.dolphin.neuralnet.conf;

import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

/**
 * Parameters for layer configuration.
 */
public final class LayerConfigurationParameters {

  @NamedParameter(doc = "Initial bias of a parameter", short_name = "initB")
  public static final class InitialBias implements Name<Double> {
  }

  @NamedParameter(doc = "Initial weight of a parameter", short_name = "initW")
  public static final class InitialWeight implements Name<Double> {
  }

  @NamedParameter(doc = "Index of the layer", short_name = "index")
  public static final class LayerIndex implements Name<Integer> {
  }

  @NamedParameter(doc = "Number of layer input nodes", short_name = "numInput")
  public static final class NumberOfInput implements Name<Integer> {
  }

  @NamedParameter(doc = "Number of layer output nodes", short_name = "numOutput")
  public static final class NumberOfOutput implements Name<Integer> {
  }

  @NamedParameter(doc = "Activation function of layer node", short_name = "activationFunc")
  public static final class ActivationFunction implements Name<String> {
  }
}
