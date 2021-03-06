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

import edu.snu.dolphin.bsp.core.DataParser;
import edu.snu.dolphin.bsp.examples.ml.parameters.MaxIterations;
import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.layerparam.provider.GroupCommParameterProvider;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.util.Validator;
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static edu.snu.dolphin.dnn.NeuralNetworkTask.*;

/**
 * Task for training a neural network when REEF Group Communication is used.
 * Compared to `NeuralNetworkTask` which terminates right after it finishes its iterations,
 * this Task waits if it finishes its iteration early and sends dummy messages to the parameter server until
 * all Tasks participating have finished the current iteration.
 * This behavior is needed due to the synchronous nature of REEF Group Communication.
 */
@TaskSide
final class GroupCommNeuralNetworkTask implements Task {
  private static final Logger LOG = Logger.getLogger(GroupCommNeuralNetworkTask.class.getName());

  private final Validator crossValidator;
  private final Validator trainingValidator;
  private final DataParser<List<Pair<Pair<Matrix, int[]>, Boolean>>> dataParser;
  private final NeuralNetwork neuralNetwork;
  private final int maxIterations;
  private final GroupCommParameterProvider parameterProvider;

  @Inject
  GroupCommNeuralNetworkTask(final DataParser<List<Pair<Pair<Matrix, int[]>, Boolean>>> dataParser,
                             final NeuralNetwork neuralNetwork,
                             @Parameter(MaxIterations.class) final int maxIterations,
                             final GroupCommParameterProvider parameterProvider) {
    this.dataParser = dataParser;
    this.neuralNetwork = neuralNetwork;
    this.maxIterations = maxIterations;
    this.trainingValidator = new Validator(neuralNetwork);
    this.crossValidator = new Validator(neuralNetwork);
    this.parameterProvider = parameterProvider;
  }

  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "GroupCommNeuralNetworkTask.call() commencing....");

    final List<Pair<Pair<Matrix, int[]>, Boolean>> dataSet = dataParser.get();
    for (int i = 0; i < maxIterations; ++i) {
      runIteration(dataSet, neuralNetwork, trainingValidator, crossValidator);

      // Send dummy messages until the parameter server sends an empty message,
      // which means all Tasks have finished their iterations.
      while (true) {
        parameterProvider.push(0, null);

        final LayerParameter[] layerParameters = parameterProvider.pull();
        if (layerParameters.length == 0) {
          break;
        }
      }

      // Aggregate validation results at the parameter server.
      parameterProvider.pushValidationStats(trainingValidator.getValidationStats(),
          crossValidator.getValidationStats());
      crossValidator.getValidationStats().reset();
      trainingValidator.getValidationStats().reset();
    }

    LOG.log(Level.INFO, "GroupCommNeuralNetworkTask.call() terminating....");
    return null;
  }
}
