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

import edu.snu.reef.dolphin.ps.ParameterServerParameters.SerializedCodecConfiguration;
import edu.snu.reef.dolphin.ps.ParameterServerParameters.SerializedUpdaterConfiguration;
import edu.snu.reef.dolphin.ps.ns.NetworkContextRegister;
import edu.snu.reef.dolphin.ps.ns.PSMessageHandler;
import edu.snu.reef.dolphin.ps.server.ServerSideMsgHandler;
import edu.snu.reef.dolphin.ps.worker.WorkerSideMsgHandler;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.evaluator.context.parameters.ContextStartHandlers;
import org.apache.reef.evaluator.context.parameters.ContextStopHandlers;
import org.apache.reef.io.network.naming.NameServer;
import org.apache.reef.io.network.naming.parameters.NameResolverNameServerAddr;
import org.apache.reef.io.network.naming.parameters.NameResolverNameServerPort;
import org.apache.reef.io.network.util.StringIdentifierFactory;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.formats.ConfigurationSerializer;
import org.apache.reef.wake.IdentifierFactory;
import org.apache.reef.wake.remote.address.LocalAddressProvider;

import javax.inject.Inject;
import java.io.IOException;

/**
 * Driver code for Parameter Server applications.
 * Should be injected into the user's Driver class.
 * Provides methods for getting context and service configurations.
 */
@DriverSide
public final class ParameterServerDriver {

  /**
   * Manager object that represents the Parameter Server configuration to be used.
   */
  private final ParameterServerManager psManager;

  /**
   * Configuration for key/preValue/value {@link org.apache.reef.wake.remote.Codec}s.
   */
  private final Configuration codecConfiguration;

  /**
   * Configuration that specifies the {@link edu.snu.reef.dolphin.ps.server.ParameterUpdater} class to use.
   */
  private final Configuration updaterConfiguration;

  /**
   * Server for providing remote identifiers to the Network Connection Service.
   */
  private final NameServer nameServer;

  /**
   * Object needed for detecting the local address of this container.
   */
  private final LocalAddressProvider localAddressProvider;

  @Inject
  private ParameterServerDriver(final ParameterServerManager psManager,
                                final ConfigurationSerializer configurationSerializer,
                                @Parameter(SerializedCodecConfiguration.class) final String serializedCodecConf,
                                @Parameter(SerializedUpdaterConfiguration.class) final String serializedUpdaterConf,
                                final NameServer nameServer,
                                final LocalAddressProvider localAddressProvider) throws IOException {
    this.psManager = psManager;
    this.codecConfiguration = configurationSerializer.fromString(serializedCodecConf);
    this.updaterConfiguration = configurationSerializer.fromString(serializedUpdaterConf);
    this.nameServer = nameServer;
    this.localAddressProvider = localAddressProvider;
  }

  /**
   * @return context configuration that both worker and server Evaluators use
   */
  private Configuration getCommonContextConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder(codecConfiguration)
        .bindSetEntry(ContextStartHandlers.class, NetworkContextRegister.RegisterContextHandler.class)
        .bindSetEntry(ContextStopHandlers.class, NetworkContextRegister.UnregisterContextHandler.class)
        .build();
  }

  /**
   * @return context configuration for an Evaluator that uses a {@code ParameterWorker}
   */
  public Configuration getWorkerContextConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder(
        getCommonContextConfiguration(),
        psManager.getWorkerContextConfiguration())
        .bindNamedParameter(PSMessageHandler.class, WorkerSideMsgHandler.class)
        .build();

  }

  /**
   * @return context configuration for an Evaluator that uses a {@code ParameterServer}
   */
  public Configuration getServerContextConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder(
        getCommonContextConfiguration(),
        psManager.getServerContextConfiguration(),
        updaterConfiguration)
        .bindNamedParameter(PSMessageHandler.class, ServerSideMsgHandler.class)
        .build();
  }

  /**
   * @return {@link NameServer} related configuration to be used by services
   */
  private Configuration getNameResolverServiceConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(NameResolverNameServerPort.class, Integer.toString(nameServer.getPort()))
        .bindNamedParameter(NameResolverNameServerAddr.class, localAddressProvider.getLocalAddress())
        .build();
  }

  /**
   * @return service configuration for an Evaluator that uses a {@code ParameterWorker}
   */
  public Configuration getWorkerServiceConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder(
        psManager.getWorkerServiceConfiguration(),
        getNameResolverServiceConfiguration())
        .bindImplementation(IdentifierFactory.class, StringIdentifierFactory.class)
        .build();
  }

  /**
   * @return service configuration for an Evaluator that uses a {@code ParameterServer}
   */
  public Configuration getServerServiceConfiguration() {
    return Tang.Factory.getTang().newConfigurationBuilder(
        psManager.getServerServiceConfiguration(),
        getNameResolverServiceConfiguration())
        .bindImplementation(IdentifierFactory.class, StringIdentifierFactory.class)
        .build();
  }
}
