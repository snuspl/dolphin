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
package edu.snu.reef.dolphin.examples.ml.algorithms.linear;

import edu.snu.reef.dolphin.core.UserParameters;
import edu.snu.reef.dolphin.examples.ml.parameters.*;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.ConfigurationBuilder;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.formats.CommandLine;

import javax.inject.Inject;

/**
 * Parameters used in svd.
 */
public final class SvdParameters implements UserParameters {
  private final int approxCnt;

  @Inject
  private SvdParameters(@Parameter(ApproxCnt.class) final int approxCnt) {
    this.approxCnt = approxCnt;
  }

  @Override
  public Configuration getDriverConf() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .build();
  }

  @Override
  public Configuration getServiceConf() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .build();
  }

  @Override
  public Configuration getUserCmpTaskConf() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(ApproxCnt.class, String.valueOf(approxCnt))
        .build();
  }

  @Override
  public Configuration getUserCtrlTaskConf() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(ApproxCnt.class, String.valueOf(approxCnt))
        .build();
  }

  public static CommandLine getCommandLine() {
    final ConfigurationBuilder cb = Tang.Factory.getTang().newConfigurationBuilder();
    final CommandLine cl = new CommandLine(cb);
    cl.registerShortNameOfClass(ApproxCnt.class);
    return cl;
  }
}
