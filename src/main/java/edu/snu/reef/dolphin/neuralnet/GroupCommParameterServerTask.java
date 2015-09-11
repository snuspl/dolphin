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

import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.SerializedLayerConfigurationSet;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.Stepsize;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.neuralnet.util.ValidationStats;
import org.apache.reef.io.network.group.api.operators.Broadcast;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.group.api.task.CommunicationGroupClient;
import org.apache.reef.io.network.group.api.task.GroupCommClient;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.apache.reef.task.Task;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import static edu.snu.reef.dolphin.neuralnet.NeuralNetworkTask.*;

/**
 * Task that acts as a parameter server for {@link GroupCommNeuralNetworkTask}s using REEF Group Communication.
 *
 * Receives activations and errors from Tasks, computes parameter gradients using those values,
 * and finally sends the updates back to the Tasks.
 */
public final class GroupCommParameterServerTask implements Task {
  private static final Logger LOG = Logger.getLogger(GroupCommParameterServerTask.class.getName());
  public static final String TASK_ID = GroupCommParameterServerTask.class.getSimpleName();

  private final LayerParameter[] layerParameters;
  private final LayerParameter[] deltaLayerParameters;
  private final float stepsize;
  private final int maxIterations;
  private final Broadcast.Sender<LayerParameter[]> layerParamBroadcastSender;
  private final Reduce.Receiver<List<Pair<List<INDArray>, List<INDArray>>>> activationErrorReduceReceiver;
  private final Reduce.Receiver<Pair<ValidationStats, ValidationStats>> validationStatsPairReduceReceiver;

  @Inject
  private GroupCommParameterServerTask(
      @Parameter(SerializedLayerConfigurationSet.class) final Set<String> serializedLayerConfigurationSet,
      @Parameter(Stepsize.class) final float stepsize,
      @Parameter(MaxIterations.class) final int maxIterations,
      final ConfigurationSerializer configurationSerializer,
      final GroupCommClient groupCommClient) {

    final CommunicationGroupClient commGroup =
        groupCommClient.getCommunicationGroup(NeuralNetworkGroupCommDriver.NeuralNetworkCommGroup.class);
    this.layerParamBroadcastSender =
        commGroup.getBroadcastSender(NeuralNetworkGroupCommDriver.LayerParamBroadcast.class);
    this.activationErrorReduceReceiver =
        commGroup.getReduceReceiver(NeuralNetworkGroupCommDriver.ActivationErrorReduce.class);
    this.validationStatsPairReduceReceiver =
        commGroup.getReduceReceiver(NeuralNetworkGroupCommDriver.ValidationStatsPairReduce.class);

    this.layerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.deltaLayerParameters = new LayerParameter[serializedLayerConfigurationSet.size()];
    this.stepsize = stepsize;
    this.maxIterations = maxIterations;

    for (final String serializedInitializerConfiguration : serializedLayerConfigurationSet) {
      try {
        final Configuration initializerConfiguration =
            configurationSerializer.fromString(serializedInitializerConfiguration);
        final LayerParameterInitializer layerParameterInitializer =
            Tang.Factory.getTang().newInjector(initializerConfiguration).getInstance(LayerParameterInitializer.class);
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
  private void resetDeltaLayerParameters() {
    for (final LayerParameter deltaLayerParameter : deltaLayerParameters) {
      deltaLayerParameter.getWeightParam().assign(0.0);
      deltaLayerParameter.getBiasParam().assign(0.0);
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
      final List<Pair<List<INDArray>, List<INDArray>>> result = activationErrorReduceReceiver.reduce();

      if (result.size() == 0) {
        // All Tasks have finished this iteration. Let's end the iteration.
        layerParamBroadcastSender.send(new LayerParameter[0]);
        final Pair<ValidationStats, ValidationStats> validationStatsPair = validationStatsPairReduceReceiver.reduce();
        LOG.log(Level.INFO,
            generateIterationLog(validationStatsPair.getFirst(), validationStatsPair.getSecond(), iteration));
        iteration++;
        continue;
      }

      // generate gradients using each pair of activations and errors
      for (final Pair<List<INDArray>, List<INDArray>> pair : result) {
        final List<INDArray> activations = pair.getFirst();
        final List<INDArray> errors = pair.getSecond();

        for (int index = 0; index < deltaLayerParameters.length; index++) {
          final INDArray activation = activations.get(index).transpose();
          final INDArray error = errors.get(index);
          deltaLayerParameters[index].getWeightParam().addi(activation.mmul(error));
          deltaLayerParameters[index].getBiasParam().addi(error);
        }
      }

      // apply the updates, regarding the size of the batch and the step size
      for (int index = 0; index < deltaLayerParameters.length; ++index) {
        final LayerParameter layerParameter = layerParameters[index];
        final LayerParameter deltaLayerParameter = deltaLayerParameters[index];
        layerParameter.getWeightParam().subi(deltaLayerParameter.getWeightParam().divi(result.size()).muli(stepsize));
        layerParameter.getBiasParam().subi(deltaLayerParameter.getBiasParam().divi(result.size()).muli(stepsize));
      }

      resetDeltaLayerParameters();
      layerParamBroadcastSender.send(layerParameters);
    }

    LOG.log(Level.INFO, "GroupCommParameterServerTask.call() terminating....");
    return null;
  }
}
