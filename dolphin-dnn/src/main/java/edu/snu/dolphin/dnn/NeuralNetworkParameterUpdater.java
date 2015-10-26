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
package edu.snu.dolphin.dnn;

import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.dolphin.dnn.data.NeuralNetParamServerData;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.util.ValidationStats;
import edu.snu.dolphin.ps.server.ParameterUpdater;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class NeuralNetworkParameterUpdater
    implements ParameterUpdater<String, NeuralNetParamServerData, NeuralNetParamServerData> {
  private static final Logger LOG = Logger.getLogger(NeuralNetworkParameterUpdater.class.getName());

  @NamedParameter(doc = "Minimum number of training example to use when outputting validation statistics",
                  short_name = "logPeriod",
                  default_value = "0")
  public final class LogPeriod implements Name<Integer> {
  }

  public static final String WHOLE_MODEL = "WHOLE_MODEL";
  public static final String VALIDATION = "VALIDATION";

  private final Set<String> serializedLayerConfigurationSet;
  private final float stepsize;
  private final ConfigurationSerializer configurationSerializer;
  private final int logPeriod;
  private int iteration;

  @Inject
  private NeuralNetworkParameterUpdater(
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(NeuralNetworkConfigurationParameters.Stepsize.class) final float stepsize,
      final ConfigurationSerializer configurationSerializer,
      @Parameter(LogPeriod.class) final int logPeriod) {
    this.serializedLayerConfigurationSet = serializedLayerConfigurationSet;
    this.stepsize = stepsize;
    this.configurationSerializer = configurationSerializer;

    if (logPeriod <= 0) {
      throw new RuntimeException("Log period is too small");
    }
    this.logPeriod = logPeriod;
    this.iteration = 0;
  }


  @Override
  public NeuralNetParamServerData process(final String key,
                                          final NeuralNetParamServerData neuralNetParamServerData) {
    if (neuralNetParamServerData.getIsValidationStatsPair()) {
      return new NeuralNetParamServerData(processValidationStats(
          neuralNetParamServerData.getValidationStatsPair().get()));
    } else {
      return new NeuralNetParamServerData(processLayerParametersList(key,
          neuralNetParamServerData.getLayerParametersList().get()));
    }
  }

  private Pair<ValidationStats, ValidationStats> processValidationStats(
      final Pair<ValidationStats, ValidationStats> validationStatsPair) {
    return validationStatsPair;
  }


  /**
   * Aggregate parameter gradients by computing the average of all gradients, per layer.
   */
  private List<LayerParameter[]> processLayerParametersList(final String key,
                                                            final List<LayerParameter[]> parameterGradientsList) {
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

    final List<LayerParameter[]> retList = new ArrayList<>(1);
    retList.add(aggregatedParameterGradients);
    return retList;
  }

  /**
   * Subtract the parameter gradients from the current layer parameter values.
   */
  @Override
  public NeuralNetParamServerData update(final NeuralNetParamServerData oldData, final NeuralNetParamServerData delta) {
    if (oldData.getIsValidationStatsPair()) {
      if (!delta.getIsValidationStatsPair()) {
        throw new RuntimeException("NeuralNetParamServerData oldData and delta have the same key but different format");
      }
      return new NeuralNetParamServerData(updateValidationStatsPair(
          oldData.getValidationStatsPair().get(), delta.getValidationStatsPair().get()));

    } else {
      if (delta.getIsValidationStatsPair()) {
        throw new RuntimeException("NeuralNetParamServerData oldData and delta have the same key but different format");
      }
      return new NeuralNetParamServerData(updateLayerParameter(
          oldData.getLayerParametersList().get().get(0), delta.getLayerParametersList().get().get(0)));
    }
  }

  private Pair<ValidationStats, ValidationStats> updateValidationStatsPair(
      final Pair<ValidationStats, ValidationStats> oldValid, final Pair<ValidationStats, ValidationStats> deltaValid) {
    final ValidationStats oldTrainingValidation = oldValid.getFirst();
    final ValidationStats deltaTrainingValidation = deltaValid.getFirst();
    final ValidationStats oldCrossValidation = oldValid.getSecond();
    final ValidationStats deltaCrossValidation = deltaValid.getSecond();

    final ValidationStats newTrainingValidation = new ValidationStats(
        oldTrainingValidation.getTotalNum() + deltaTrainingValidation.getTotalNum(),
        oldTrainingValidation.getCorrectNum() + deltaTrainingValidation.getCorrectNum());
    final ValidationStats newCrossValidation = new ValidationStats(
        oldCrossValidation.getTotalNum() + deltaCrossValidation.getTotalNum(),
        oldCrossValidation.getCorrectNum() + deltaCrossValidation.getCorrectNum());

    if (oldTrainingValidation.getTotalNum() + deltaTrainingValidation.getTotalNum() >= logPeriod) {
      LOG.log(Level.INFO,
          NeuralNetworkTask.generateIterationLog(newTrainingValidation, newCrossValidation, iteration++));
      newTrainingValidation.reset();
      newCrossValidation.reset();
    }

    return new Pair<>(newTrainingValidation, newCrossValidation);
  }

  private List<LayerParameter[]> updateLayerParameter(final LayerParameter[] layerParameters,
                                                      final LayerParameter[] parameterGradients) {
    for (int index = 0; index < layerParameters.length; ++index) {
      final LayerParameter layerParameter = layerParameters[index];
      final LayerParameter parameterGradient = parameterGradients[index];
      layerParameter.getWeightParam().subi(parameterGradient.getWeightParam());
      layerParameter.getBiasParam().subi(parameterGradient.getBiasParam());
    }

    final List<LayerParameter[]> retList = new ArrayList<>(1);
    retList.add(layerParameters);
    return retList;
  }

  /**
   * Use {@link LayerParameterInitializer} to generate initial layer parameter values.
   */
  @Override
  public NeuralNetParamServerData initValue(final String key) {
    if (key.equals(WHOLE_MODEL)) {
      return new NeuralNetParamServerData(initValueLayerParameters());
    } else if (key.equals(VALIDATION)) {
      return new NeuralNetParamServerData(initValueValidationStatsPair());
    } else {
      throw new RuntimeException("Unexpected key: " + key);
    }
  }

  private Pair<ValidationStats, ValidationStats> initValueValidationStatsPair() {
    return new Pair<>(new ValidationStats(), new ValidationStats());
  }

  private List<LayerParameter[]> initValueLayerParameters() {
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

    final List<LayerParameter[]> retList = new ArrayList<>(1);
    retList.add(layerParameters);
    return retList;
  }
}
