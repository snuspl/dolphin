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
package edu.snu.dolphin.dnn.conf;

import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

/**
 * Parameters for layer configuration.
 *
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

  @NamedParameter(doc = "the shape of input data for layer")
  public static final class LayerInputShape implements Name<String> {
  }

  /**
   * For fully connected layers.
   */
  @NamedParameter(doc = "number of layer output nodes", short_name = "numOutput")
  public static final class NumberOfOutput implements Name<Integer> {
  }

  /**
   * For activation layers.
   */
  @NamedParameter(doc = "activation function of layer node", short_name = "activationFunc")
  public static final class ActivationFunction implements Name<String> {
  }

  @NamedParameter(doc = "loss function of loss layer", short_name = "lossFunc")
  public static final class LossFunction implements Name<String> {
  }

  /**
   * For pooling layers.
   */
  @NamedParameter(doc = "pooling type of pooling layer")
  public static final class PoolingType implements Name<String> {
  }

  @NamedParameter(doc = "stride height of pooling / convolutional layer", short_name = "strideH")
  public static final class StrideHeight implements Name<Integer> {
  }

  @NamedParameter(doc = "stride width of pooling / convolutional layer", short_name = "strideW")
  public static final class StrideWidth implements Name<Integer> {
  }

  @NamedParameter(doc = "kernel height of pooling / convolutional layer", short_name = "kernelH")
  public static final class KernelHeight implements Name<Integer> {
  }

  @NamedParameter(doc = "kernel width of pooling / convolutional layer", short_name = "kernelW")
  public static final class KernelWidth implements Name<Integer> {
  }
}
