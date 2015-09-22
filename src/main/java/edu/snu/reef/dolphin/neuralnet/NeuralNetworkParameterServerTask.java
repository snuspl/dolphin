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

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.examples.ml.parameters.MaxIterations;
import edu.snu.reef.dolphin.neuralnet.data.NeuralNetParamWorkerData;
import edu.snu.reef.dolphin.neuralnet.util.Validator;
import edu.snu.reef.dolphin.ps.worker.ParameterWorker;
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static edu.snu.reef.dolphin.neuralnet.NeuralNetworkTask.*;

/**
 * The task that trains neural network with the data set.
 *
 * Assumes the input file can be parsed by NeuralNetworkDataParser.
 */
@TaskSide
public final class NeuralNetworkParameterServerTask implements Task {
  private static final Logger LOG = Logger.getLogger(NeuralNetworkParameterServerTask.class.getName());

  private final Validator crossValidator;
  private final Validator trainingValidator;
  private final DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> dataParser;
  private final NeuralNetwork neuralNetwork;
  private final int maxIterations;
  private final ParameterWorker<String, NeuralNetParamWorkerData, ?> worker;

  @Inject
  NeuralNetworkParameterServerTask(final DataParser<List<Pair<Pair<INDArray, Integer>, Boolean>>> dataParser,
                                   final NeuralNetwork neuralNetwork,
                                   @Parameter(MaxIterations.class) final int maxIterations,
                                   final ParameterWorker<String, NeuralNetParamWorkerData, ?> worker) {
    super();
    this.dataParser = dataParser;
    this.neuralNetwork = neuralNetwork;
    this.maxIterations = maxIterations;
    this.trainingValidator = new Validator(neuralNetwork);
    this.crossValidator = new Validator(neuralNetwork);
    this.worker = worker;
  }

  /** {@inheritDoc} */
  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "ComputeTask.call() commencing....");

    final List<Pair<Pair<INDArray, Integer>, Boolean>> dataSet = dataParser.get();
    for (int i = 0; i < maxIterations; ++i) {
      runIteration(dataSet, neuralNetwork, trainingValidator, crossValidator);

      worker.push(NeuralNetworkParameterUpdater.VALIDATION, new NeuralNetParamWorkerData(
          new Pair<>(trainingValidator.getValidationStats(), crossValidator.getValidationStats())));

      crossValidator.getValidationStats().reset();
      trainingValidator.getValidationStats().reset();
    }

    return null;
  }
}
