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

import edu.snu.dolphin.bsp.examples.ml.parameters.MaxIterations;
import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.Stepsize;
import edu.snu.dolphin.dnn.layerparam.initializer.LayerParameterInitializer;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.util.ValidationStats;
import org.apache.reef.io.network.group.api.operators.Broadcast;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.group.api.task.CommunicationGroupClient;
import org.apache.reef.io.network.group.api.task.GroupCommClient;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.apache.reef.task.Task;

import javax.inject.Inject;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import static edu.snu.dolphin.dnn.NeuralNetworkTask.*;

/**
 * Task that acts as a parameter server for {@link GroupCommNeuralNetworkTask}s using REEF Group Communication.
 * <p/>
 * Receives parameter gradients from Tasks, computes updated parameters using those values,
 * and finally sends the updates back to the Tasks.
 */
public final class GroupCommParameterServerTask implements Task {
  private static final Logger LOG = Logger.getLogger(GroupCommParameterServerTask.class.getName());
  public static final String TASK_ID = GroupCommParameterServerTask.class.getSimpleName();

  private final MatrixFactory matrixFactory;
  private final LayerParameter[] layerParameters;
  private final LayerParameter[] deltaLayerParameters;
  private final float stepsize;
  private final int maxIterations;
  private final Broadcast.Sender<LayerParameter[]> layerParamBroadcastSender;
  private final Reduce.Receiver<List<Pair<Integer, LayerParameter[]>>> parameterGradientReduceReceiver;
  private final Reduce.Receiver<Pair<ValidationStats, ValidationStats>> validationStatsPairReduceReceiver;

  @Inject
  private GroupCommParameterServerTask(
      final MatrixFactory matrixFactory,
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(Stepsize.class) final float stepsize,
      @Parameter(MaxIterations.class) final int maxIterations,
      final ConfigurationSerializer configurationSerializer,
      final GroupCommClient groupCommClient,
      final Injector injector) {

    final CommunicationGroupClient commGroup =
        groupCommClient.getCommunicationGroup(NeuralNetworkGroupCommDriver.NeuralNetworkCommGroup.class);
    this.layerParamBroadcastSender =
        commGroup.getBroadcastSender(NeuralNetworkGroupCommDriver.LayerParamBroadcast.class);
    this.parameterGradientReduceReceiver =
        commGroup.getReduceReceiver(NeuralNetworkGroupCommDriver.ParameterGradientReduce.class);
    this.validationStatsPairReduceReceiver =
        commGroup.getReduceReceiver(NeuralNetworkGroupCommDriver.ValidationStatsPairReduce.class);

    this.matrixFactory = matrixFactory;
    this.layerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.deltaLayerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.stepsize = stepsize;
    this.maxIterations = maxIterations;

    for (final String serializedInitializerConfiguration : serializedLayerConfigurationSet) {
      try {
        final Configuration initializerConfiguration =
            configurationSerializer.fromString(serializedInitializerConfiguration);
        final LayerParameterInitializer layerParameterInitializer =
            injector.forkInjector(initializerConfiguration).getInstance(LayerParameterInitializer.class);
        final int index = layerParameterInitializer.getIndex();

        this.layerParameters[index] = layerParameterInitializer.generateInitialParameter();

      } catch (final IOException e) {
        throw new RuntimeException("IOException while deserializing configuration", e);
      } catch (final InjectionException e) {
        throw new RuntimeException("InjectionException while injecting LayerParameterInitializer", e);
      }
    }

    initDeltaParameters();
  }

  /**
   * Initializes delta parameters with zero matrices.
   */
  private void initDeltaParameters() {
    for (int i = 0; i < layerParameters.length; ++i) {
      final Matrix biasParam = layerParameters[i].getBiasParam();
      final Matrix weightParam = layerParameters[i].getWeightParam();

      deltaLayerParameters[i] = LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.zeros(weightParam.getRows(), weightParam.getColumns()))
          .setBiasParam(matrixFactory.zeros(biasParam.getRows(), biasParam.getColumns()))
          .build();
    }
  }

  /**
   * Resets the number of updates and delta parameters by filling zeros.
   */
  private void resetDeltaLayerParameters() {
    for (final LayerParameter deltaLayerParameter : deltaLayerParameters) {
      deltaLayerParameter.getWeightParam().fill(0.0f);
      deltaLayerParameter.getBiasParam().fill(0.0f);
    }
  }

  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "GroupCommParameterServerTask.call() commencing....");
    long loopIndex = 0;
    int iteration = 0;

    // The variable `iteration` does not indicate the number of times this while loop has ran.
    // Rather, `iteration` tracks the number of iterations `GroupCommNeuralNetworkTask`s have finished up until now.
    while (iteration < maxIterations) {
      LOG.log(Level.INFO, "GroupCommParameterServerTask.call() loop {0}....", loopIndex++);
      final List<Pair<Integer, LayerParameter[]>> result = parameterGradientReduceReceiver.reduce();

      if (result.size() == 0) {
        // All Tasks have finished this iteration. Let's end the iteration.
        layerParamBroadcastSender.send(new LayerParameter[0]);
        final Pair<ValidationStats, ValidationStats> validationStatsPair = validationStatsPairReduceReceiver.reduce();
        LOG.log(Level.INFO,
            generateIterationLog(validationStatsPair.getFirst(), validationStatsPair.getSecond(), iteration));
        iteration++;
        continue;
      }

      // aggregate parameter gradients
      int batchSizeSum = 0;
      for (final Pair<Integer, LayerParameter[]> intAndParameterGradientPair : result) {
        batchSizeSum += intAndParameterGradientPair.getFirst();
        final LayerParameter[] parameterGradient = intAndParameterGradientPair.getSecond();
        for (int index = 0; index < deltaLayerParameters.length; index++) {
          deltaLayerParameters[index].getWeightParam().addi(parameterGradient[index].getWeightParam());
          deltaLayerParameters[index].getBiasParam().addi(parameterGradient[index].getBiasParam());
        }
      }

      // apply the updates, regarding the size of the batch and the step size
      for (int index = 0; index < deltaLayerParameters.length; ++index) {
        final LayerParameter layerParameter = layerParameters[index];
        final LayerParameter deltaLayerParameter = deltaLayerParameters[index];
        final float factor = stepsize / batchSizeSum;
        layerParameter.getWeightParam().subi(deltaLayerParameter.getWeightParam().muli(factor));
        layerParameter.getBiasParam().subi(deltaLayerParameter.getBiasParam().muli(factor));
      }

      resetDeltaLayerParameters();
      layerParamBroadcastSender.send(layerParameters);
    }

    LOG.log(Level.INFO, "GroupCommParameterServerTask.call() terminating....");
    return null;
  }
}
