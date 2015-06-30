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
package edu.snu.reef.dolphin.core;

import com.microsoft.reef.io.network.group.operators.Broadcast;
import com.microsoft.reef.io.network.nggroup.api.task.CommunicationGroupClient;
import com.microsoft.reef.io.network.nggroup.api.task.GroupCommClient;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataGatherSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataScatterReceiver;
import edu.snu.reef.dolphin.groupcomm.names.*;
import org.apache.reef.driver.task.TaskConfigurationOptions;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.task.HeartBeatTriggerManager;
import org.apache.reef.task.Task;
import org.apache.reef.task.TaskMessage;
import org.apache.reef.task.TaskMessageSource;
import org.apache.reef.util.Optional;
import org.apache.reef.wake.remote.impl.ObjectSerializableCodec;

import javax.inject.Inject;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class for Compute Task.
 * Compute Task, which is a part of a stage,
 * specifies local computation using data stored in local storage.
 */
public final class ComputeTask implements Task, TaskMessageSource {
  public static final String TASK_ID_PREFIX = "CmpTask";
  private static final Logger LOG = Logger.getLogger(ComputeTask.class.getName());

  private final String taskId;
  private final UserComputeTask userComputeTask;
  private final CommunicationGroupClient commGroup;
  private final HeartBeatTriggerManager heartBeatTriggerManager;
  private final Broadcast.Receiver<CtrlMessage> ctrlMessageBroadcast;
  private final ObjectSerializableCodec<Long> codecLong = new ObjectSerializableCodec<>();
  private long runTime = -1;

  /**
   * Compute Task constructor - instantiated via TANG.
   *
   * @param taskId id of the current task
   * @param commGroupName
   * @param groupCommClient
   * @param userComputeTask
   * @param heartBeatTriggerManager
   * @throws ClassNotFoundException
   */
  @Inject
  private ComputeTask(@Parameter(TaskConfigurationOptions.Identifier.class) final String taskId,
                     @Parameter(CommunicationGroup.class) final String commGroupName,
                     final GroupCommClient groupCommClient,
                     final UserComputeTask userComputeTask,
                     final HeartBeatTriggerManager heartBeatTriggerManager) throws ClassNotFoundException {
    this.taskId = taskId;
    this.commGroup = groupCommClient.getCommunicationGroup(
        (Class<? extends Name<String>>) Class.forName(commGroupName));
    this.ctrlMessageBroadcast = commGroup.getBroadcastReceiver(CtrlMsgBroadcast.class);
    this.userComputeTask = userComputeTask;
    this.heartBeatTriggerManager = heartBeatTriggerManager;
  }

  @Override
  public byte[] call(final byte[] memento) throws Exception {
    LOG.log(Level.INFO, String.format("%s starting...", taskId));

    userComputeTask.initialize();
    int iteration=0;
    while (!isTerminated()) {
      receiveData(iteration);
      final long runStart = System.currentTimeMillis();
      userComputeTask.run(iteration);
      runTime = System.currentTimeMillis() - runStart;
      sendData(iteration);
      heartBeatTriggerManager.triggerHeartBeat();
      iteration++;
    }
    userComputeTask.cleanup();

    return null;
  }

  /**
   * Receive data from Controller Task.
   *
   * @param iteration the current number of iterations
   * @throws Exception
   */
  private void receiveData(int iteration) throws Exception {
    if (userComputeTask.isBroadcastUsed()) {
      ((DataBroadcastReceiver)userComputeTask).receiveBroadcastData(iteration,
          commGroup.getBroadcastReceiver(DataBroadcast.class).receive());
    }
    if (userComputeTask.isScatterUsed()) {
      ((DataScatterReceiver)userComputeTask).receiveScatterData(iteration,
          commGroup.getScatterReceiver(DataScatter.class).receive());
    }
  }

  /**
   * Send data to Controller Task.
   *
   * @param iteration the current number of iterations
   * @throws Exception
   */
  private void sendData(int iteration) throws Exception {
    if (userComputeTask.isGatherUsed()) {
      commGroup.getGatherSender(DataGather.class).send(
          ((DataGatherSender)userComputeTask).sendGatherData(iteration));
    }
    if (userComputeTask.isReduceUsed()) {
      commGroup.getReduceSender(DataReduce.class).send(
          ((DataReduceSender)userComputeTask).sendReduceData(iteration));
    }
  }

  /**
   * Check whether Controller Task asks Compute Tasks to terminate or not.
   *
   * @return whether to terminate or not
   * @throws Exception
   */
  private boolean isTerminated() throws Exception {
    return ctrlMessageBroadcast.receive() == CtrlMessage.TERMINATE;
  }

  @Override
  public synchronized Optional<TaskMessage> getMessage() {
    return Optional.of(TaskMessage.from(ComputeTask.class.getName(),
        this.codecLong.encode(this.runTime)));
  }
}
