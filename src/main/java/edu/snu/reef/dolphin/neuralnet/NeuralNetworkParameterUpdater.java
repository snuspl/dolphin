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
package edu.snu.reef.dolphin.neuralnet;

import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.ps.server.ParameterUpdater;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.io.IOException;
import java.util.List;
import java.util.Set;

public final class NeuralNetworkParameterUpdater
    implements ParameterUpdater<String, List<LayerParameter[]>, LayerParameter[]> {

  public static final String WHOLE_MODEL = "WHOLE_MODEL";

  private final Set<String> serializedLayerConfigurationSet;
  private final float stepsize;
  private final ConfigurationSerializer configurationSerializer;

  @Inject
  private NeuralNetworkParameterUpdater(
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(NeuralNetworkConfigurationParameters.Stepsize.class) final float stepsize,
      final ConfigurationSerializer configurationSerializer) {
    this.serializedLayerConfigurationSet = serializedLayerConfigurationSet;
    this.stepsize = stepsize;
    this.configurationSerializer = configurationSerializer;
  }

  /**
   * Aggregate parameter gradients by computing the average of all gradients, per layer.
   */
  @Override
  public LayerParameter[] process(final String key, final List<LayerParameter[]> parameterGradientsList) {
    if (parameterGradientsList == null || parameterGradientsList.size() == 0 || !key.equals(WHOLE_MODEL)) {
      return null;
    }

    final LayerParameter[] aggregatedParameterGradients = new LayerParameter[parameterGradientsList.get(0).length];
    for (int index = 0; index < aggregatedParameterGradients.length; index++) {
      aggregatedParameterGradients[index] = LayerParameter.newBuilder()
          .setWeightParam(Nd4j.zeros(parameterGradientsList.get(0)[index].getWeightParam().shape()))
          .setBiasParam(Nd4j.zeros(parameterGradientsList.get(0)[index].getBiasParam().shape()))
          .build();
    }

    for (final LayerParameter[] parameterGradient : parameterGradientsList) {
      for (int index = 0; index < aggregatedParameterGradients.length; index++) {
        aggregatedParameterGradients[index].getWeightParam().addi(parameterGradient[index].getWeightParam());
        aggregatedParameterGradients[index].getBiasParam().addi(parameterGradient[index].getBiasParam());
      }
    }

    for (final LayerParameter sumParameterGradient : aggregatedParameterGradients) {
      sumParameterGradient.getWeightParam().divi(parameterGradientsList.size()).muli(stepsize);
      sumParameterGradient.getBiasParam().divi(parameterGradientsList.size()).muli(stepsize);
    }

    return aggregatedParameterGradients;
  }

  /**
   * Subtract the parameter gradients from the current layer parameter values.
   */
  @Override
  public LayerParameter[] update(final LayerParameter[] layerParameters, final LayerParameter[] parameterGradients) {
    for (int index = 0; index < layerParameters.length; ++index) {
      final LayerParameter layerParameter = layerParameters[index];
      final LayerParameter parameterGradient = parameterGradients[index];
      layerParameter.getWeightParam().subi(parameterGradient.getWeightParam());
      layerParameter.getBiasParam().subi(parameterGradient.getBiasParam());
    }

    return layerParameters;
  }

  /**
   * Use {@link LayerParameterInitializer} to generate initial layer parameter values.
   */
  @Override
  public LayerParameter[] initValue(final String key) {
    if (!key.equals(WHOLE_MODEL)) {
      throw new RuntimeException("Unexpected key: " + key);
    }

    final LayerParameter[] layerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];

    for (final String serializedInitializerConfiguration : serializedLayerConfigurationSet) {
      try {
        final Configuration initializerConfiguration =
            configurationSerializer.fromString(serializedInitializerConfiguration);
        final Injector injector = Tang.Factory.getTang().newInjector(initializerConfiguration);
        final LayerParameterInitializer layerParameterInitializer =
            injector.getInstance(LayerParameterInitializer.class);
        final int index = layerParameterInitializer.getIndex();

        layerParameters[index] = layerParameterInitializer.generateInitialParameter();

      } catch (final IOException exception) {
        throw new RuntimeException("IOException during de-serializing layer configuration", exception);
      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException during injecting LayerParameterInitializer", exception);
      }
    }

    return layerParameters;
  }
}
