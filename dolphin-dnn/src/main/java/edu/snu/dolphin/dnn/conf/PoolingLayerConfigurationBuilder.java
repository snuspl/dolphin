/*
 * Copyright (C) 2016 Seoul National University
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

import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layerparam.initializer.PoolingLayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.PoolingLayer;
import edu.snu.dolphin.dnn.layers.LayerBase;
import edu.snu.dolphin.dnn.proto.NeuralNetworkProtos;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.util.Builder;

/**
 * Configuration builder for Pooling layer.
 *
 * The configuration that this builder generates is used to create a pooling layer instance.
 * The generate configuration need to bind the parameter for a layer input shape, to inject layer instance.
 */
public final class PoolingLayerConfigurationBuilder implements Builder<Configuration> {

  public static PoolingLayerConfigurationBuilder newConfigurationBuilder() {
    return new PoolingLayerConfigurationBuilder();
  }

  private String poolingType;
  private int strideHeight;
  private int strideWidth;
  private int kernelHeight;
  private int kernelWidth;

  public synchronized PoolingLayerConfigurationBuilder setPoolingType(final String poolingType) {
    this.poolingType = poolingType;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setStrideHeight(final int strideHeight) {
    this.strideHeight = strideHeight;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setStrideWidth(final int strideWidth) {
    this.strideWidth = strideWidth;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setKernelHeight(final int kernelHeight) {
    this.kernelHeight = kernelHeight;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder setKernelWidth(final int kernelWidth) {
    this.kernelWidth = kernelWidth;
    return this;
  }

  public synchronized PoolingLayerConfigurationBuilder fromProtoConfiguration(
      final NeuralNetworkProtos.LayerConfiguration protoConf) {
    poolingType = protoConf.getPoolingParam().getPoolingType();
    strideHeight = protoConf.getPoolingParam().getStrideHeight();
    strideWidth = protoConf.getPoolingParam().getStrideWidth();
    kernelHeight = protoConf.getPoolingParam().getKernelHeight();
    kernelWidth = protoConf.getPoolingParam().getKernelWidth();
    return this;
  }

  @Override
  public synchronized Configuration build() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(LayerConfigurationParameters.PoolingType.class, poolingType)
        .bindNamedParameter(LayerConfigurationParameters.StrideHeight.class, String.valueOf(strideHeight))
        .bindNamedParameter(LayerConfigurationParameters.StrideWidth.class, String.valueOf(strideWidth))
        .bindNamedParameter(LayerConfigurationParameters.KernelHeight.class, String.valueOf(kernelHeight))
        .bindNamedParameter(LayerConfigurationParameters.KernelWidth.class, String.valueOf(kernelWidth))
        .bindImplementation(LayerBase.class, PoolingLayer.class)
        .bindImplementation(LayerParameterInitializer.class, PoolingLayerParameterInitializer.class)
        .build();
  }
}
