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
import edu.snu.dolphin.dnn.util.ValidationStats;
import edu.snu.dolphin.dnn.util.Validator;
import org.apache.reef.annotations.audience.TaskSide;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;

import javax.inject.Inject;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The task that trains neural network with the data set.
 *
 * Assumes the input file can be parsed by NeuralNetworkDataParser.
 */
@TaskSide
public final class NeuralNetworkTask implements Task {

  private static final Logger LOG = Logger.getLogger(NeuralNetworkTask.class.getName());
  private static final String NEWLINE = System.getProperty("line.separator");

  private final Validator crossValidator;
  private final Validator trainingValidator;
  private final DataParser<List<Pair<Pair<Matrix, Integer>, Boolean>>> dataParser;
  private final NeuralNetwork neuralNetwork;
  private final int maxIterations;

  @Inject
  NeuralNetworkTask(final DataParser<List<Pair<Pair<Matrix, Integer>, Boolean>>> dataParser,
                    final NeuralNetwork neuralNetwork,
                    @Parameter(MaxIterations.class) final int maxIterations) {
    super();
    this.dataParser = dataParser;
    this.neuralNetwork = neuralNetwork;
    this.maxIterations = maxIterations;
    this.trainingValidator = new Validator(neuralNetwork);
    this.crossValidator = new Validator(neuralNetwork);
  }

  /** {@inheritDoc} */
  @Override
  public byte[] call(final byte[] bytes) throws Exception {
    LOG.log(Level.INFO, "ComputeTask.call() commencing....");

    final List<Pair<Pair<Matrix, Integer>, Boolean>> dataSet = dataParser.get();
    for (int i = 0; i < maxIterations; ++i) {
      runIteration(dataSet, neuralNetwork, trainingValidator, crossValidator);
      LOG.log(Level.INFO, generateIterationLog(trainingValidator.getValidationStats(),
          crossValidator.getValidationStats(), i));

      crossValidator.getValidationStats().reset();
      trainingValidator.getValidationStats().reset();
    }

    return null;
  }

  public static void runIteration(final List<Pair<Pair<Matrix, Integer>, Boolean>> dataSet,
                                  final NeuralNetwork neuralNetwork,
                                  final Validator trainingValidator,
                                  final Validator crossValidator) {
    for (final Pair<Pair<Matrix, Integer>, Boolean> data : dataSet) {
      final Matrix input = data.getFirst().getFirst();
      final int label = data.getFirst().getSecond();
      final boolean isValidation = data.getSecond();
      if (isValidation) {
        crossValidator.validate(input, label);
      } else {
        neuralNetwork.train(input, label);
        trainingValidator.validate(input, label);
      }
    }
  }

  public static String generateIterationLog(final ValidationStats trainingValidationStats,
                                            final ValidationStats crossValidationStats,
                                            final int iteration) {
    return new StringBuilder()
        .append(NEWLINE)
        .append("=========================================================")
        .append(NEWLINE)
        .append("Iteration: ")
        .append(iteration)
        .append(NEWLINE)
        .append("Training Error: ")
        .append(trainingValidationStats.getError())
        .append(NEWLINE)
        .append("Cross Validation Error: ")
        .append(crossValidationStats.getError())
        .append(NEWLINE)
        .append("# of training inputs: ")
        .append(trainingValidationStats.getTotalNum())
        .append(NEWLINE)
        .append("# of validation inputs: ")
        .append(crossValidationStats.getTotalNum())
        .append(NEWLINE)
        .append("=========================================================")
        .append(NEWLINE)
        .toString();

  }
}
