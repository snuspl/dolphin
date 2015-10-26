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

import edu.snu.dolphin.bsp.core.DataParseService;
import edu.snu.dolphin.dnn.NeuralNetworkParameterUpdater.LogPeriod;
import edu.snu.dolphin.dnn.data.NeuralNetworkDataParser;
import edu.snu.dolphin.ps.driver.ParameterServerDriver;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.driver.context.ActiveContext;
import org.apache.reef.driver.context.ContextConfiguration;
import org.apache.reef.driver.task.CompletedTask;
import org.apache.reef.driver.task.TaskConfiguration;
import org.apache.reef.io.data.loading.api.DataLoadingService;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.annotations.Unit;
import org.apache.reef.wake.EventHandler;

import javax.inject.Inject;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The driver code for the neural network REEF application that uses REEF Parameter Server.
 */
@DriverSide
@Unit
public final class NeuralNetworkSingleNodeParameterServerDriver {
  private static final Logger LOG = Logger.getLogger(NeuralNetworkDriver.class.getName());
  private static final String NNCONTEXT_PREFIX = "NeuralNetworkContext-";
  private static final String NNTASK_PREFIX = "NeuralNetworkTask-";

  private final AtomicInteger neuralNetworkContextIds = new AtomicInteger();
  private final AtomicInteger neuralNetworkTaskIds = new AtomicInteger();
  private final NeuralNetworkESParameters neuralNetworkESParameters;
  private final AtomicInteger neuralNetworkTaskCount;
  private ActiveContext serverContext;
  private final int logPeriod;

  /**
   * Accessor for Data Loading Service.
   * Can check whether a evaluator is configured with the service or not.
   */
  private final DataLoadingService dataLoadingService;

  /**
   * Accessor for Parameter Server service.
   * Provides context and service configurations.
   */
  private final ParameterServerDriver psDriver;

  @Inject
  private NeuralNetworkSingleNodeParameterServerDriver(final DataLoadingService dataLoadingService,
                                                       final NeuralNetworkESParameters neuralNetworkESParameters,
                                                       final ParameterServerDriver psDriver,
                                                       @Parameter(LogPeriod.class) final int logPeriod) {
    this.dataLoadingService = dataLoadingService;
    this.neuralNetworkESParameters = neuralNetworkESParameters;
    this.psDriver = psDriver;
    this.neuralNetworkTaskCount = new AtomicInteger(dataLoadingService.getNumberOfPartitions());
    this.logPeriod = logPeriod;
  }

  final class ActiveContextHandler implements EventHandler<ActiveContext> {

    /** {@inheritDoc} */
    @Override
    public void onNext(final ActiveContext activeContext) {
      final String contextId = activeContext.getId();
      LOG.log(Level.FINER, "Context active: {0}", contextId);

      // Case 1: Evaluator configured with no input data.
      // We add a parameter server context and service to it.
      if (dataLoadingService.isComputeContext(activeContext)) {
        final Configuration contextConf = Configurations.merge(
            ContextConfiguration.CONF
                .set(ContextConfiguration.IDENTIFIER, "ParameterServerContext")
                .build(),
            psDriver.getServerContextConfiguration());
        final Configuration serviceConf = Configurations.merge(
            psDriver.getServerServiceConfiguration(),
            neuralNetworkESParameters.getServiceAndNeuralNetworkConfiguration(),
            Tang.Factory.getTang().newConfigurationBuilder()
                .bindNamedParameter(LogPeriod.class, String.valueOf(logPeriod))
                .build());

        LOG.log(Level.FINEST, "Submit parameter server context {0} to {1}", new Object[]{activeContext, contextId});
        activeContext.submitContextAndService(contextConf, serviceConf);

      // Case 2: Evaluator configured with a Data Loading context has been given.
      // We need to add a neural network context above this context.
      } else if (dataLoadingService.isDataLoadedContext(activeContext)) {
        final String nnCtxtId = NNCONTEXT_PREFIX + neuralNetworkContextIds.getAndIncrement();
        final Configuration contextConf = Configurations.merge(
            ContextConfiguration.CONF.set(ContextConfiguration.IDENTIFIER, nnCtxtId).build(),
            psDriver.getWorkerContextConfiguration());

        // Add Data Parse Service and Neural Network Service configurations
        final Configuration serviceConf = Configurations.merge(
            DataParseService.getServiceConfiguration(NeuralNetworkDataParser.class),
            neuralNetworkESParameters.getServiceConfiguration(),
            psDriver.getWorkerServiceConfiguration());

        LOG.log(Level.FINEST, "Submit neural network context {0} to {1}", new Object[]{nnCtxtId, contextId});
        activeContext.submitContextAndService(contextConf, serviceConf);

      // Case 3: Evaluator configured with a neural network context.
      // We can now place a neural network task on top of the contexts.
      } else if (activeContext.getId().startsWith(NNCONTEXT_PREFIX)) {
        final String taskId = NNTASK_PREFIX + neuralNetworkTaskIds.getAndIncrement();

        LOG.log(Level.FINEST, "Submit neural network task {0} to {1}", new Object[]{taskId, contextId});
        activeContext.submitTask(Configurations.merge(
            TaskConfiguration.CONF
                .set(TaskConfiguration.IDENTIFIER, taskId)
                .set(TaskConfiguration.TASK, ParameterServerNeuralNetworkTask.class)
                .build(),
            neuralNetworkESParameters.getTaskConfiguration()));

      // Case 4: Parameter Server Context has been given.
      // We simply save the activeContext value to a private field for later use.
      } else {
        serverContext = activeContext;
      }
    }
  }

  final class CompletedTaskHandler implements EventHandler<CompletedTask> {
    @Override
    public void onNext(final CompletedTask completedTask) {
      completedTask.getActiveContext().close();
      if (neuralNetworkTaskCount.decrementAndGet() <= 0 && serverContext != null) {
        // shut down the server when all tasks have completed
        serverContext.close();
      }
    }
  }
}
