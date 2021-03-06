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
package edu.snu.dolphin.bsp.core;

import edu.snu.dolphin.bsp.core.metric.MetricManager;
import edu.snu.dolphin.bsp.core.metric.MetricTracker;
import edu.snu.dolphin.bsp.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.dolphin.bsp.groupcomm.interfaces.DataGatherReceiver;
import edu.snu.dolphin.bsp.groupcomm.interfaces.DataReduceReceiver;
import edu.snu.dolphin.bsp.groupcomm.interfaces.DataScatterSender;
import edu.snu.dolphin.bsp.groupcomm.names.*;
import edu.snu.dolphin.bsp.core.metric.MetricTrackers;
import org.apache.reef.driver.task.TaskConfigurationOptions;
import org.apache.reef.io.network.group.api.operators.Broadcast;
import org.apache.reef.io.network.group.api.task.CommunicationGroupClient;
import org.apache.reef.io.network.group.api.task.GroupCommClient;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.Task;

import javax.inject.Inject;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class ControllerTask implements Task {
  public static final String TASK_ID_PREFIX = "CtrlTask";
  private static final Logger LOG = Logger.getLogger(ControllerTask.class.getName());

  private final String taskId;
  private final UserControllerTask userControllerTask;
  private final CommunicationGroupClient commGroup;
  private final Broadcast.Sender<CtrlMessage> ctrlMessageBroadcast;
  private final MetricManager metricManager;
  private final Set<MetricTracker> metricTrackerSet;

  @Inject
  public ControllerTask(final GroupCommClient groupCommClient,
                        final UserControllerTask userControllerTask,
                        @Parameter(TaskConfigurationOptions.Identifier.class) final String taskId,
                        @Parameter(CommunicationGroup.class) final String commGroupName,
                        final MetricManager metricManager,
                        @Parameter(MetricTrackers.class) final Set<MetricTracker> metricTrackerSet)
      throws ClassNotFoundException {
    this.commGroup =
        groupCommClient.getCommunicationGroup((Class<? extends Name<String>>) Class.forName(commGroupName));
    this.userControllerTask = userControllerTask;
    this.taskId = taskId;
    this.ctrlMessageBroadcast = commGroup.getBroadcastSender(CtrlMsgBroadcast.class);
    this.metricManager = metricManager;
    this.metricTrackerSet = metricTrackerSet;
  }

  @Override
  public byte[] call(final byte[] memento) throws Exception {
    LOG.log(Level.INFO, String.format("%s starting...", taskId));

    userControllerTask.initialize();
    try (final MetricManager metricManager = this.metricManager;) {
      metricManager.registerTrackers(metricTrackerSet);
      int iteration = 0;
      while (!userControllerTask.isTerminated(iteration)) {
        metricManager.start();
        ctrlMessageBroadcast.send(CtrlMessage.RUN);
        sendData(iteration);
        receiveData(iteration);
        userControllerTask.run(iteration);
        metricManager.stop();
        updateTopology();
        iteration++;
      }
      ctrlMessageBroadcast.send(CtrlMessage.TERMINATE);
      userControllerTask.cleanup();
    }

    return null;
  }

  /**
   * Update the group communication topology, if it has changed.
   */
  private void updateTopology() {
    if (commGroup.getTopologyChanges().exist()) {
      commGroup.updateTopology();
    }
  }

  private void sendData(final int iteration) throws Exception {
    if (userControllerTask.isBroadcastUsed()) {
      commGroup.getBroadcastSender(DataBroadcast.class).send(
          ((DataBroadcastSender) userControllerTask).sendBroadcastData(iteration));
    }
    if (userControllerTask.isScatterUsed()) {
      commGroup.getScatterSender(DataScatter.class).send(
          ((DataScatterSender) userControllerTask).sendScatterData(iteration));
    }
  }

  private void receiveData(final int iteration) throws Exception {
    if (userControllerTask.isGatherUsed()) {
      ((DataGatherReceiver)userControllerTask).receiveGatherData(iteration,
          commGroup.getGatherReceiver(DataGather.class).receive());
    }
    if (userControllerTask.isReduceUsed()) {
      ((DataReduceReceiver)userControllerTask).receiveReduceData(iteration,
          commGroup.getReduceReceiver(DataReduce.class).reduce());
    }
  }
}
