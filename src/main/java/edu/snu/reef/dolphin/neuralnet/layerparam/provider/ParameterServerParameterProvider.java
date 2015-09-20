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

import edu.snu.reef.dolphin.neuralnet.NeuralNetworkParameterUpdater;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationParameters.BatchSize;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.ps.worker.ParameterWorker;
import org.apache.reef.tang.annotations.Parameter;

import javax.annotation.concurrent.ThreadSafe;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.List;

/**
 * Parameter provider for a neural network that uses REEF Parameter Server.
 * <p/>
 * Sends parameter gradients to the server using a certain batch size.
 * Receives updated parameters from the server.
 */
@ThreadSafe
public final class ParameterServerParameterProvider implements ParameterProvider {

  private static final int RETRY_COUNT = 3;

  private final ParameterWorker<String, List<LayerParameter[]>, LayerParameter[]> worker;
  private final List<LayerParameter[]> parameterGradientsList;
  private final int batchSize;
  private int pushCount;

  @Inject
  private ParameterServerParameterProvider(
      final ParameterWorker<String, List<LayerParameter[]>, LayerParameter[]> worker,
      @Parameter(BatchSize.class) final int batchSize) {
    this.worker = worker;
    this.parameterGradientsList = new ArrayList<>(batchSize);
    this.batchSize = batchSize;
    this.pushCount = 0;
  }

  @Override
  public synchronized void push(final LayerParameter[] parameterGradients) {
    parameterGradientsList.add(parameterGradients);

    if (++pushCount > batchSize) {
      pushCount = 0;
      worker.push(NeuralNetworkParameterUpdater.WHOLE_MODEL, parameterGradientsList);
      parameterGradientsList.clear();
    }
  }

  @Override
  public LayerParameter[] pull() {
    int retryCount = 0;
    while (retryCount < RETRY_COUNT) {
      final LayerParameter[] retVal = worker.pull(NeuralNetworkParameterUpdater.WHOLE_MODEL);
      if (retVal != null) {
        return retVal;
      }

      retryCount++;
    }

    throw new RuntimeException("Retried " + RETRY_COUNT + " times but failed to pull model from server.");
  }
}
