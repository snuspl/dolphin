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

import edu.snu.dolphin.dnn.NeuralNetworkParameterUpdater;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationParameters.BatchSize;
import edu.snu.dolphin.dnn.data.NeuralNetParamServerData;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.ps.worker.ParameterWorker;
import org.apache.reef.tang.annotations.Parameter;

import javax.annotation.concurrent.ThreadSafe;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.List;

/**
 * Parameter provider for a neural network that uses Dolphin Parameter Server, {@code dolphin-ps}.
 * <p/>
 * Sends parameter gradients to the server using a certain batch size.
 * Receives updated parameters from the server.
 */
@ThreadSafe
public final class ParameterServerParameterProvider implements ParameterProvider {

  private static final int RETRY_COUNT = 3;

  private final ParameterWorker<String, NeuralNetParamServerData, NeuralNetParamServerData> worker;
  private final List<LayerParameter[]> parameterGradientsList;
  private final int batchSize;
  private int pushCount;

  @Inject
  private ParameterServerParameterProvider(
      final ParameterWorker<String, NeuralNetParamServerData, NeuralNetParamServerData> worker,
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
      worker.push(NeuralNetworkParameterUpdater.WHOLE_MODEL, new NeuralNetParamServerData(parameterGradientsList));
      parameterGradientsList.clear();
    }
  }

  @Override
  public LayerParameter[] pull() {
    int retryCount = 0;
    while (retryCount < RETRY_COUNT) {
      final NeuralNetParamServerData neuralNetParamServerData = worker.pull(NeuralNetworkParameterUpdater.WHOLE_MODEL);
      if (neuralNetParamServerData == null) {
        retryCount++;
        continue;
      }

      if (neuralNetParamServerData.isValidationStatsPair()) {
        throw new RuntimeException("Requested NeuralNetworkParameterUpdater.WHOLE_MODEL but received validation stats");
      }

      final List<LayerParameter[]> retList = neuralNetParamServerData.getLayerParametersList();
      if (retList.size() != 1) {
        throw new RuntimeException(String.format("Expected one array of LayerParameters but received %d arrays",
            retList.size()));
      }

      return retList.get(0);
    }

    throw new RuntimeException("Retried " + RETRY_COUNT + " times but failed to pull model from server.");
  }
}
