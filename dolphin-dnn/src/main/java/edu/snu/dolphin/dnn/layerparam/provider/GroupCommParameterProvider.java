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

import edu.snu.dolphin.dnn.NeuralNetworkGroupCommDriver;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.util.ValidationStats;
import org.apache.reef.exception.evaluator.NetworkException;
import org.apache.reef.io.network.group.api.operators.Broadcast;
import org.apache.reef.io.network.group.api.operators.Reduce;
import org.apache.reef.io.network.group.api.task.CommunicationGroupClient;
import org.apache.reef.io.network.group.api.task.GroupCommClient;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;

import javax.annotation.concurrent.ThreadSafe;
import javax.inject.Inject;
import java.util.*;

/**
 * Parameter provider for a neural network that uses REEF Group Communication.
 * <p/>
 * Sends parameter gradients to the server with a certain batch size.
 * Receives updated parameters from the server.
 */
@ThreadSafe
public final class GroupCommParameterProvider implements ParameterProvider {

  private final List<LayerParameter[]> parameterGradientList;
  private final int batchSize;
  private final Broadcast.Receiver<LayerParameter[]> layerParamBroadcastReceiver;
  private final Reduce.Sender<List<LayerParameter[]>> parameterGradientReduceSender;
  private final Reduce.Sender<Pair<ValidationStats, ValidationStats>> validationStatsReduceSender;
  private int pushCount;

  @Inject
  private GroupCommParameterProvider(
      @Parameter(NeuralNetworkConfigurationParameters.BatchSize.class) final int batchSize,
      final GroupCommClient groupCommClient) {

    this.parameterGradientList = new ArrayList<>(batchSize);
    this.batchSize = batchSize;
    this.pushCount = 0;

    final CommunicationGroupClient commGroup =
        groupCommClient.getCommunicationGroup(NeuralNetworkGroupCommDriver.NeuralNetworkCommGroup.class);
    this.layerParamBroadcastReceiver =
        commGroup.getBroadcastReceiver(NeuralNetworkGroupCommDriver.LayerParamBroadcast.class);
    this.parameterGradientReduceSender =
        commGroup.getReduceSender(NeuralNetworkGroupCommDriver.ParameterGradientReduce.class);
    this.validationStatsReduceSender =
        commGroup.getReduceSender(NeuralNetworkGroupCommDriver.ValidationStatsPairReduce.class);
  }

  @Override
  public synchronized void push(final LayerParameter[] parameterGradients) {
    // do not store the input if it is not valid
    if (!(parameterGradients == null || parameterGradients.length == 0)) {
      parameterGradientList.add(parameterGradients);
    }

    if (++pushCount >= batchSize) {
      pushCount = 0;

      try {
        parameterGradientReduceSender.send(parameterGradientList);
      } catch (final NetworkException e) {
        throw new RuntimeException("NetworkException while trying to send reduce", e);
      } catch (final InterruptedException e) {
        throw new RuntimeException("InterruptedException while trying to send reduce", e);
      }
      parameterGradientList.clear();
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
