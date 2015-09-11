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
import edu.snu.reef.dolphin.neuralnet.data.*;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.driver.context.ActiveContext;
import org.apache.reef.driver.task.TaskConfiguration;
import org.apache.reef.evaluator.context.parameters.ContextIdentifier;
import org.apache.reef.io.data.loading.api.DataLoadingService;
import org.apache.reef.io.network.group.api.driver.CommunicationGroupDriver;
import org.apache.reef.io.network.group.api.driver.GroupCommDriver;
import org.apache.reef.io.network.group.impl.config.BroadcastOperatorSpec;
import org.apache.reef.io.network.group.impl.config.ReduceOperatorSpec;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;
import org.apache.reef.tang.annotations.Unit;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.wake.EventHandler;

import javax.inject.Inject;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The driver code for the neural network REEF application when using REEF Group Communication.
 */
@DriverSide
@Unit
public final class NeuralNetworkGroupCommDriver {
  private static final Logger LOG = Logger.getLogger(NeuralNetworkGroupCommDriver.class.getName());

  private final AtomicInteger taskIds = new AtomicInteger();
  private final NeuralNetworkTaskParameters neuralNetworkTaskParameters;

  /**
   * Accessor for Data Loading Service.
   * Can check whether an evaluator is configured with the service or not.
   */
  private final DataLoadingService dataLoadingService;

  /**
   * Accessor for Group Communication Service.
   * Add communication groups using this object.
   */
  private final GroupCommDriver groupCommDriver;

  /**
   * The single communication group we will use for this application.
   * For this particular app, we use Broadcast and Reduce.
   */
  private final CommunicationGroupDriver commGroup;

  @Inject
  private NeuralNetworkGroupCommDriver(final DataLoadingService dataLoadingService,
                                       final NeuralNetworkTaskParameters neuralNetworkTaskParameters,
                                       final GroupCommDriver groupCommDriver) {
    this.dataLoadingService = dataLoadingService;
    this.neuralNetworkTaskParameters = neuralNetworkTaskParameters;
    this.groupCommDriver = groupCommDriver;
    this.commGroup =
        groupCommDriver.newCommunicationGroup(NeuralNetworkCommGroup.class,
            dataLoadingService.getNumberOfPartitions() + 1);
    this.commGroup
        .addBroadcast(LayerParamBroadcast.class,
            BroadcastOperatorSpec.newBuilder()
                .setDataCodecClass(LayerParameterArrayCodec.class)
                .setSenderId(GroupCommParameterServerTask.TASK_ID)
                .build())
        .addReduce(ActivationErrorReduce.class,
            ReduceOperatorSpec.newBuilder()
                .setDataCodecClass(ActivationErrorListCodec.class)
                .setReduceFunctionClass(ListReduceFunction.class)
                .setReceiverId(GroupCommParameterServerTask.TASK_ID)
                .build())
        .addReduce(ValidationStatsPairReduce.class,
            ReduceOperatorSpec.newBuilder()
                .setDataCodecClass(ValidationStatsPairCodec.class)
                .setReduceFunctionClass(ValidationStatsPairReduceFunction.class)
                .setReceiverId(GroupCommParameterServerTask.TASK_ID)
                .build())
        .finalise();

  }

  final class ActiveContextHandler implements EventHandler<ActiveContext> {
    /**
     * String for detecting the context configured for the controller task, `GroupCommParameterServerTask`.
     */
    private String ctrlTaskCtxtId;

    private String getContextId(final Configuration contextConf) {
      try {
        final Injector injector = Tang.Factory.getTang().newInjector(contextConf);
        return injector.getNamedInstance(ContextIdentifier.class);
      } catch (final InjectionException e) {
        throw new RuntimeException("Unable to inject context identifier from context conf", e);
      }
    }

    @Override
    public void onNext(final ActiveContext activeContext) {
      final String contextId = activeContext.getId();
      LOG.log(Level.FINER, "Context active: {0}", contextId);

      // Case 1: Evaluator configured with no input data.
      // We need to add a group comm context for the group comm parameter server above this context.
      if (dataLoadingService.isComputeContext(activeContext)) {
        final Configuration groupCommContextConf = groupCommDriver.getContextConfiguration();
        final Configuration groupCommServiceConf = groupCommDriver.getServiceConfiguration();
        final String nnCtrlCtxtId = getContextId(groupCommContextConf);
        ctrlTaskCtxtId = nnCtrlCtxtId;
        LOG.log(Level.FINEST, "Submit group comm parameter server context {0} to {1}",
            new Object[]{nnCtrlCtxtId, contextId});

        activeContext.submitContextAndService(groupCommContextConf, groupCommServiceConf);

      // Case 2: Evaluator configured with a Data Loading context has been given.
      // We need to add a group comm context for training neural networks above this context.
      } else if (dataLoadingService.isDataLoadedContext(activeContext)) {
        final Configuration groupCommContextConf = groupCommDriver.getContextConfiguration();
        final Configuration groupCommServiceConf = groupCommDriver.getServiceConfiguration();
        final String nnCmpCtxtId = getContextId(groupCommContextConf);
        LOG.log(Level.FINEST, "Submit group comm neural network context {0} to {1}",
            new Object[]{nnCmpCtxtId, contextId});

        // Add Data Parse Service
        final Configuration dataParseConf = DataParseService.getServiceConfiguration(NeuralNetworkDataParser.class);

        activeContext.submitContextAndService(groupCommContextConf,
            Configurations.merge(groupCommServiceConf, dataParseConf));

      // Case 3: Evaluator configured with a group comm parameter server context.
      // We can now place a group comm parameter server task on top of this context.
      } else if (contextId.equals(ctrlTaskCtxtId)) {
        final String taskId = GroupCommParameterServerTask.TASK_ID;
        LOG.log(Level.FINEST, "Submit group comm parameter server task {0} to {1}", new Object[]{taskId, contextId});

        final Configuration partialTaskConf = Configurations.merge(
            TaskConfiguration.CONF
                .set(TaskConfiguration.IDENTIFIER, taskId)
                .set(TaskConfiguration.TASK, GroupCommParameterServerTask.class)
                .build(),
            neuralNetworkTaskParameters.getTaskConfiguration());
        commGroup.addTask(partialTaskConf);
        activeContext.submitTask(groupCommDriver.getTaskConfiguration(partialTaskConf));

      // Case 4: Evaluator configured with a group comm neural network context.
      // We can now place a neural network task on top of this context.
      } else {
        final String taskId = "nnTask-" + taskIds.getAndIncrement();
        LOG.log(Level.FINEST, "Submit neural network task {0} to {1}", new Object[]{taskId, contextId});

        final Configuration partialTaskConf = Configurations.merge(
            TaskConfiguration.CONF
                .set(TaskConfiguration.IDENTIFIER, taskId)
                .set(TaskConfiguration.TASK, GroupCommNeuralNetworkTask.class)
                .build(),
            neuralNetworkTaskParameters.getTaskConfiguration());
        commGroup.addTask(partialTaskConf);
        activeContext.submitTask(groupCommDriver.getTaskConfiguration(partialTaskConf));
      }
    }
  }

  @NamedParameter(doc = "Name of the communication group used for the Neural Network application")
  public final class NeuralNetworkCommGroup implements Name<String> {
  }

  @NamedParameter(doc = "Name of the Broadcast operator used in the Neural Network application")
  public final class LayerParamBroadcast implements Name<String> {
  }

  @NamedParameter(doc = "Name of the Reduce operator used for aggregating network activations and errors")
  public final class ActivationErrorReduce implements Name<String> {
  }

  @NamedParameter(doc = "Name of the Reduce operator used for aggregating validation results")
  public final class ValidationStatsPairReduce implements Name<String> {
  }
}
