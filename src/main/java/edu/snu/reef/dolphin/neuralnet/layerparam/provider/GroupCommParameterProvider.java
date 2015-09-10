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

import edu.snu.reef.dolphin.neuralnet.NeuralNetworkGroupCommDriver;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.neuralnet.util.ValidationStats;
import org.apache.reef.exception.evaluator.NetworkException;
import org.apache.reef.io.network.group.api.operators.Broadcast;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.group.api.task.CommunicationGroupClient;
import org.apache.reef.io.network.group.api.task.GroupCommClient;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.annotation.concurrent.ThreadSafe;
import javax.inject.Inject;
import java.util.*;

/**
 * Parameter provider for a neural network that uses REEF Group Communication.
 *
 * Sends activation values and gradients to the server with a certain batch size.
 * Receives parameter updates from the server.
 */
@ThreadSafe
public final class GroupCommParameterProvider implements ParameterProvider {

  private final List<Pair<List<INDArray>, List<INDArray>>> activationGradientList;
  private final int batchSize;
  private final Broadcast.Receiver<LayerParameter[]> layerParamBroadcastReceiver;
  private final Reduce.Sender<List<Pair<List<INDArray>, List<INDArray>>>> activationGradientReduceSender;
  private final Reduce.Sender<Pair<ValidationStats, ValidationStats>> validationStatsReduceSender;
  private int pushCount;

  @Inject
  private GroupCommParameterProvider(
      @Parameter(NeuralNetworkConfigurationParameters.BatchSize.class) final int batchSize,
      final GroupCommClient groupCommClient) {

    this.activationGradientList = Collections.synchronizedList(
        new ArrayList<Pair<List<INDArray>, List<INDArray>>>(batchSize));
    this.batchSize = batchSize;
    this.pushCount = 0;

    final CommunicationGroupClient commGroup =
        groupCommClient.getCommunicationGroup(NeuralNetworkGroupCommDriver.NeuralNetworkCommGroup.class);
    this.layerParamBroadcastReceiver =
        commGroup.getBroadcastReceiver(NeuralNetworkGroupCommDriver.LayerParamBroadcast.class);
    this.activationGradientReduceSender =
        commGroup.getReduceSender(NeuralNetworkGroupCommDriver.ActivationGradientReduce.class);
    this.validationStatsReduceSender =
        commGroup.getReduceSender(NeuralNetworkGroupCommDriver.ValidationStatsPairReduce.class);
  }

  @Override
  public synchronized void push(final List<INDArray> activations, final List<INDArray> gradients) {
    // do not store the input if it is not valid
    if (!(activations == null || activations.size() == 0 || gradients == null || gradients.size() == 0)) {
      activationGradientList.add(new Pair<>(activations, gradients));
    }

    if (++pushCount >= batchSize) {
      pushCount = 0;

      try {
        activationGradientReduceSender.send(activationGradientList);
      } catch (final NetworkException e) {
        throw new RuntimeException("NetworkException while trying to send reduce", e);
      } catch (final InterruptedException e) {
        throw new RuntimeException("InterruptedException while trying to send reduce", e);
      }
      activationGradientList.clear();
    }

  }

  @Override
  public synchronized LayerParameter[] pull() {
    try {
      return layerParamBroadcastReceiver.receive();

    } catch (final NetworkException e) {
      throw new RuntimeException("NetworkException while trying to receive broadcast", e);
    } catch (final InterruptedException e) {
      throw new RuntimeException("InterruptedException while trying to receive broadcast", e);
    }
  }

  public synchronized void pushValidationStats(final ValidationStats trainingValidationStats,
                                               final ValidationStats crossValidationStats) {
    try {
      validationStatsReduceSender.send(new Pair<>(trainingValidationStats, crossValidationStats));
    } catch (final NetworkException e) {
      throw new RuntimeException("NetworkException while trying to send reduce", e);
    } catch (final InterruptedException e) {
      throw new RuntimeException("InterruptedException while trying to send reduce", e);
    }
  }
}
