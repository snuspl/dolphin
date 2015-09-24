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
package edu.snu.reef.dolphin.ps.driver;

import edu.snu.reef.dolphin.ps.ns.EndpointId;
import edu.snu.reef.dolphin.ps.server.ParameterServer;
import edu.snu.reef.dolphin.ps.server.SingleNodeParameterServer;
import edu.snu.reef.dolphin.ps.worker.ParameterWorker;
import edu.snu.reef.dolphin.ps.worker.SingleNodeParameterWorker;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.driver.context.ServiceConfiguration;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

import javax.inject.Inject;

/**
 * Manager class for a Parameter Server that uses only one node for a server.
 * This manager does NOT handle server or worker faults.
 */
@DriverSide
public final class SingleNodeParameterServerManager implements ParameterServerManager {
  private static final String SERVER_ID = "SINGLE_NODE_SERVER_ID";

  @Inject
  private SingleNodeParameterServerManager() {
  }

  /**
   * Returns worker-side context configuration.
   * Binds the server id to a named parameter so that workers know where to send push/pull messages to.
   */
  @Override
  public Configuration getWorkerContextConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(ServerId.class, SERVER_ID)
        .build();
  }

  /**
   * Returns server-side context configuration.
   * Binds the server id to a named parameter so that
   * the server uses it to register itself to Network Connection Service.
   */
  @Override
  public Configuration getServerContextConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(EndpointId.class, SERVER_ID)
        .build();
  }

  /**
   * Returns worker-side service configuration.
   * Sets {@link SingleNodeParameterWorker} as the {@link ParameterWorker} class.
   */
  @Override
  public Configuration getWorkerServiceConfiguration() {
    return Tang.Factory.getTang()
        .newConfigurationBuilder(ServiceConfiguration.CONF
            .set(ServiceConfiguration.SERVICES, SingleNodeParameterWorker.class)
            .build())
        .bindImplementation(ParameterWorker.class, SingleNodeParameterWorker.class)
        .build();
  }

  /**
   * Returns server-side service configuration.
   * Sets {@link SingleNodeParameterServer} as the {@link ParameterWorker} class.
   */
  @Override
  public Configuration getServerServiceConfiguration() {
    return Tang.Factory.getTang()
        .newConfigurationBuilder(ServiceConfiguration.CONF
            .set(ServiceConfiguration.SERVICES, SingleNodeParameterServer.class)
            .build())
        .bindImplementation(ParameterServer.class, SingleNodeParameterServer.class)
        .build();
  }

  @NamedParameter(doc = "server identifier for Network Connection Service")
  public final class ServerId implements Name<String> {
  }
}
