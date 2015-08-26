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
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.GroupCommParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.ParameterProvider;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;

import javax.inject.Inject;
import java.util.logging.Logger;

/**
 * A wrapper Task for `NeuralNetworkTask` when REEF Group Communication is used.
 * Instead of `NeuralNetworkTask` which terminates right after it finished its iterations,
 * this Task waits and sends dummy messages to the parameter server until all Tasks participating have finished.
 * This behavior is needed due to the synchronous nature of REEF Group Communication.
 */
@TaskSide
public final class GroupCommNeuralNetworkTask implements Task {
  private static final Logger LOG = Logger.getLogger(GroupCommNeuralNetworkTask.class.getName());

  private final int batchSize;
  private final ParameterProvider parameterProvider;
  private final NeuralNetworkTask neuralNetworkTask;

  @Inject
  GroupCommNeuralNetworkTask(@Parameter(NeuralNetworkConfigurationParameters.BatchSize.class) final int batchSize,
                             final GroupCommParameterProvider parameterProvider,
                             final NeuralNetworkTask neuralNetworkTask) {
    this.batchSize = batchSize;
    this.parameterProvider = parameterProvider;
    this.neuralNetworkTask = neuralNetworkTask;
  }

  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    neuralNetworkTask.call(bytes);

    // Send dummy messages until the parameter server sends an empty message,
    // which means all Tasks have finished their iterations.
    while (true) {
      for (int index = 0; index < batchSize; index++) {
        parameterProvider.push(null, null);
      }

      final LayerParameter[] layerParameters = parameterProvider.pull();
      if (layerParameters.length == 0) {
        break;
      }
    }

    return null;
  }
}
