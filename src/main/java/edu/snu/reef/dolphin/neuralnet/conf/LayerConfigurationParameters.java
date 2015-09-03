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
 * <p/>
 * When we need new parameters of new type of layer, we can add them as inner classes of this class.
 */
public final class LayerConfigurationParameters {

  @NamedParameter(doc = "initial bias of a parameter", short_name = "initB")
  public static final class InitialBias implements Name<Float> {
  }

  @NamedParameter(
      doc = "standard deviation of a normal distribution that is used to generate initial weight of a parameter",
      short_name = "initW")
  public static final class InitialWeight implements Name<Float> {
  }

  @NamedParameter(doc = "random seed that is used to generate initial weight", short_name = "seed")
  public static final class RandomSeed implements Name<Long> {
  }

  @NamedParameter(doc = "index of the layer", short_name = "index")
  public static final class LayerIndex implements Name<Integer> {
  }

  @NamedParameter(doc = "number of layer input nodes", short_name = "numInput")
  public static final class NumberOfInput implements Name<Integer> {
  }

  @NamedParameter(doc = "number of layer output nodes", short_name = "numOutput")
  public static final class NumberOfOutput implements Name<Integer> {
  }

  @NamedParameter(doc = "activation function of layer node", short_name = "activationFunc")
  public static final class ActivationFunction implements Name<String> {
  }

  @NamedParameter(doc = "kernel size of pooling layer node", short_name = "kernelSize")
  public static final class KernelSize implements Name<Integer> {
  }

  @NamedParameter(doc = "pooling function of pooling layer node", short_name = "poolingFunc", default_value = "max")
  public static final class PoolingFunction implements Name<String> {
  }
}
