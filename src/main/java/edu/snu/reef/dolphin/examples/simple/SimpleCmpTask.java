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

import edu.snu.reef.dolphin.core.UserComputeTask;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.dolphin.groupcomm.interfaces.DataReduceSender;

import javax.inject.Inject;

/**
 * User-defined compute task class for the simple example application.
 * At every iteration, compute tasks print out the given message.
 */
public final class SimpleCmpTask extends UserComputeTask
    implements DataBroadcastReceiver<String>, DataReduceSender<Integer> {

  /**
   * message to print.
   */
  private String message = null;

  /**
   * number of times that the task prints out the message.
   */
  private Integer count = 0;

  @Inject
  private SimpleCmpTask() {
  }

  @Override
  public void run(int iteration) {
    System.out.println(message);
    count++;

  }

  @Override
  public void receiveBroadcastData(int iteration, String data) {
    message = data;
    count = 0;
  }

  @Override
  public Integer sendReduceData(int iteration) {
    return count;
  }
}