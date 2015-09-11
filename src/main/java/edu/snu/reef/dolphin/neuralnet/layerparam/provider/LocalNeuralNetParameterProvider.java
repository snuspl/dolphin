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

import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.Stepsize;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.io.IOException;
import java.util.Set;

/**
 * Parameter provider for a neural network on the local environment.
 * <p/>
 * Calculates the updated parameters by adding the average of parameter gradients.
 */
public final class LocalNeuralNetParameterProvider implements ParameterProvider {

  private final LayerParameter[] layerParameters;
  private final LayerParameter[] deltaLayerParameters;
  private final float stepsize;
  private int numUpdate = 0;

  @Inject
  public LocalNeuralNetParameterProvider(
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(Stepsize.class) final float stepsize,
      final ConfigurationSerializer configurationSerializer) {
    this.layerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.deltaLayerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.stepsize = stepsize;

    for (final String serializedInitializerConfiguration : serializedLayerConfigurationSet) {
      try {
        final Configuration initializerConfiguration =
            configurationSerializer.fromString(serializedInitializerConfiguration);
        final Injector injector = Tang.Factory.getTang().newInjector(initializerConfiguration);
        final LayerParameterInitializer layerParameterInitializer =
            injector.getInstance(LayerParameterInitializer.class);
        final int index = layerParameterInitializer.getIndex();

        this.layerParameters[index] = layerParameterInitializer.generateInitialParameter();

      } catch (final IOException exception) {
        throw new RuntimeException("IOException", exception);
      } catch (final InjectionException exception) {
        throw new RuntimeException("InjectionException", exception);
      }
    }
    initDeltaParameters();
  }

  /**
   * Initializes delta parameters with zero matrices.
   */
  private void initDeltaParameters() {
    for (int i = 0; i < layerParameters.length; ++i) {
      final INDArray biasParam = layerParameters[i].getBiasParam();
      final INDArray weightParam = layerParameters[i].getWeightParam();

      deltaLayerParameters[i] = LayerParameter.newBuilder()
          .setWeightParam(Nd4j.zeros(weightParam.shape()))
          .setBiasParam(Nd4j.zeros(biasParam.shape()))
          .build();
    }
  }

  /**
   * Resets the number of updates and delta parameters by filling zeros.
   */
  private void reset() {
    for (final LayerParameter layerParameter : deltaLayerParameters) {
      layerParameter.getWeightParam().assign(0.0);
      layerParameter.getBiasParam().assign(0.0);
    }
    numUpdate = 0;
  }

  /** {@inheritDoc} */
  @Override
  public void push(final LayerParameter[] parameterGradients) {
    assert parameterGradients.length == deltaLayerParameters.length;
    for (int i = 0; i < layerParameters.length; ++i) {
      deltaLayerParameters[i].getWeightParam().addi(parameterGradients[i].getWeightParam());
      deltaLayerParameters[i].getBiasParam().addi(parameterGradients[i].getBiasParam());
    }
    ++numUpdate;
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter[] pull() {
    if (numUpdate > 0) {
      for (int i = 0; i < deltaLayerParameters.length; ++i) {
        final LayerParameter layerParameter = layerParameters[i];
        final LayerParameter deltaLayerParameter = deltaLayerParameters[i];
        layerParameter.getWeightParam().subi(deltaLayerParameter.getWeightParam().divi(numUpdate).muli(stepsize));
        layerParameter.getBiasParam().subi(deltaLayerParameter.getBiasParam().divi(numUpdate).muli(stepsize));
      }
      reset();
    }
    return layerParameters;
  }
}
