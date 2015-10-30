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

import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.Stepsize;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;

import javax.inject.Inject;
import java.io.IOException;
import java.util.Set;

/**
 * Parameter provider for a neural network on the local environment.
 * <p/>
 * Calculates the updated parameters by stochastic gradient descent algorithm.
 */
public final class LocalNeuralNetParameterProvider implements ParameterProvider {

  private final LayerParameter[] layerParameters;
  private final float stepsize;

  @Inject
  public LocalNeuralNetParameterProvider(
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(Stepsize.class) final float stepsize,
      final ConfigurationSerializer configurationSerializer,
      final Injector injector) {
    this.layerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.stepsize = stepsize;

    for (final String serializedInitializerConfiguration : serializedLayerConfigurationSet) {
      try {
        final Configuration initializerConfiguration =
            configurationSerializer.fromString(serializedInitializerConfiguration);
        final LayerParameterInitializer layerParameterInitializer =
            injector.forkInjector(initializerConfiguration).getInstance(LayerParameterInitializer.class);
        final int index = layerParameterInitializer.getIndex();

        this.layerParameters[index] = layerParameterInitializer.generateInitialParameter();

      } catch (final IOException exception) {
        throw new RuntimeException("IOException", exception);
      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException", exception);
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public void push(final int batchSize, final LayerParameter[] parameterGradients) {
    final float factor = stepsize / batchSize;
    for (int i = 0; i < layerParameters.length; ++i) {
      layerParameters[i].getWeightParam().subi(parameterGradients[i].getWeightParam().mul(factor));
      layerParameters[i].getBiasParam().subi(parameterGradients[i].getBiasParam().mul(factor));
    }
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter[] pull() {
    return layerParameters;
  }
}
