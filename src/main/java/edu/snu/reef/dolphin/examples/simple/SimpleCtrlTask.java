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
package edu.snu.reef.dolphin.examples.simple;

import edu.snu.reef.dolphin.core.UserControllerTask;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastSender;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceReceiver;

import javax.inject.Inject;

/**
 * User-defined controller task class for the simple example application.
 * It broadcast message across compute tasks
 * and reduce the number of times that compute tasks print out the message.
 */
public final class SimpleCtrlTask extends UserControllerTask
    implements DataReduceReceiver<Integer>, DataBroadcastSender<String> {

  /**
   * Number of times that compute tasks print out the message sent by the controller task.
   */
  private Integer count = 0;

  @Inject
  private SimpleCtrlTask() {
  }

  @Override
  public void run(int iteration) {
    System.out.println(String.format("Number of Tasks Printing Message: %d", count));
  }

  @Override
  public boolean isTerminated(int iteration) {
    return iteration >= 10;
  }

  @Override
  public String sendBroadcastData(int iteration) {
    return String.format("Hello, REEF!: %d", iteration);
  }

  @Override
  public void receiveReduceData(int iteration, Integer data) {
    this.count = data;
  }
}
