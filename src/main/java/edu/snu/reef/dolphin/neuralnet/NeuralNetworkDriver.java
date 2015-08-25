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

import edu.snu.reef.dolphin.core.DataParseService;
import edu.snu.reef.dolphin.neuralnet.data.NeuralNetworkDataParser;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.driver.context.ActiveContext;
import org.apache.reef.driver.context.ContextConfiguration;
import org.apache.reef.driver.task.TaskConfiguration;
import org.apache.reef.io.data.loading.api.DataLoadingService;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.annotations.Unit;
import org.apache.reef.tang.exceptions.BindException;
import org.apache.reef.wake.EventHandler;

import javax.inject.Inject;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The driver code for the neural network REEF application.
 */
@DriverSide
@Unit
public final class NeuralNetworkDriver {

  private static final Logger LOG = Logger.getLogger(NeuralNetworkDriver.class.getName());

  private final AtomicInteger ctrlCtxIds = new AtomicInteger();
  private final AtomicInteger taskIds = new AtomicInteger();
  private final NeuralNetworkTaskParameters neuralNetworkTaskParameters;

  /**
   * Accessor for data loading service
   * Can check whether a evaluator is configured with the service or not.
   */
  private final DataLoadingService dataLoadingService;

  @Inject
  private NeuralNetworkDriver(final DataLoadingService dataLoadingService,
                              final NeuralNetworkTaskParameters neuralNetworkTaskParameters) {
    this.dataLoadingService = dataLoadingService;
    this.neuralNetworkTaskParameters = neuralNetworkTaskParameters;
  }

  final class ActiveContextHandler implements EventHandler<ActiveContext> {

    /** {@inheritDoc} */
    @Override
    public void onNext(final ActiveContext activeContext) {
      final String contextId = activeContext.getId();
      LOG.log(Level.FINER, "Context active: {0}", contextId);


      // Case 1: Evaluator configured with a Data Loading context has been given.
      // We need to add a neural network context above this context.
      if (dataLoadingService.isDataLoadedContext(activeContext)) {
        final String nnCtxtId = "nnCtxt-" + ctrlCtxIds.getAndIncrement();
        LOG.log(Level.FINEST, "Submit neural network context {0} to: {1}",
            new Object[]{nnCtxtId, contextId});

        // Add a Data Parse service
        final Configuration dataParseConf = DataParseService.getServiceConfiguration(NeuralNetworkDataParser.class);

        activeContext.submitContextAndService(
            ContextConfiguration.CONF.set(ContextConfiguration.IDENTIFIER, nnCtxtId).build(),
            dataParseConf);

        // Case 2: Evaluator configured with a neural network context.
        // We can now place a neural network task on top of the contexts.
      } else if (activeContext.getId().startsWith("nnCtxt")) {
        final String taskId = "nnTask-" + taskIds.getAndIncrement();
        LOG.log(Level.FINEST, "Submit neural network task {0} to :{1}", new Object[]{taskId, contextId});

        try {
          activeContext.submitTask(Configurations.merge(
              TaskConfiguration.CONF
                  .set(TaskConfiguration.IDENTIFIER, taskId)
                  .set(TaskConfiguration.TASK, NeuralNetworkTask.class)
                  .build(),
              neuralNetworkTaskParameters.getTaskConfiguration()));
        } catch (final BindException ex) {
          LOG.log(Level.SEVERE, "Configuration error in " + contextId, ex);
          throw new RuntimeException("Configuration error in " + contextId, ex);
        }
      } else {
        LOG.log(Level.FINEST, "Neural network Task {0} -- Closing", contextId);
        activeContext.close();
      }
    }
  }
}
