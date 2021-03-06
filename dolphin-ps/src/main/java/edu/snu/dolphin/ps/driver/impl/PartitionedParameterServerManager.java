/*
 * Copyright (C) 2016 Seoul National University
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
package edu.snu.dolphin.ps.driver.impl;

import edu.snu.dolphin.ps.driver.api.ParameterServerManager;
import edu.snu.dolphin.ps.ns.EndpointId;
import edu.snu.dolphin.ps.ns.PSMessageHandler;
import edu.snu.dolphin.ps.server.partitioned.PartitionedParameterServer;
import edu.snu.dolphin.ps.server.partitioned.PartitionedServerSideMsgHandler;
import edu.snu.dolphin.ps.server.partitioned.PartitionedServerSideReplySender;
import edu.snu.dolphin.ps.server.partitioned.PartitionedServerSideReplySenderImpl;
import edu.snu.dolphin.ps.common.partitioned.parameters.NumServers;
import edu.snu.dolphin.ps.common.partitioned.parameters.NumPartitions;
import edu.snu.dolphin.ps.server.partitioned.parameters.ServerQueueSize;
import edu.snu.dolphin.ps.worker.AsyncWorkerHandler;
import edu.snu.dolphin.ps.worker.api.ParameterWorker;
import edu.snu.dolphin.ps.worker.partitioned.ContextStopHandler;
import edu.snu.dolphin.ps.worker.partitioned.PartitionedParameterWorker;
import edu.snu.dolphin.ps.worker.partitioned.PartitionedWorkerHandler;
import edu.snu.dolphin.ps.common.partitioned.resolver.ServerResolver;
import edu.snu.dolphin.ps.common.partitioned.resolver.StaticServerResolver;
import org.apache.reef.annotations.audience.DriverSide;
import org.apache.reef.driver.context.ServiceConfiguration;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.concurrent.atomic.AtomicInteger;

import static edu.snu.dolphin.ps.common.Constants.SERVER_ID_PREFIX;
import static edu.snu.dolphin.ps.common.Constants.WORKER_ID_PREFIX;

/**
 * Manager class for a Partitioned Parameter Server, that supports atomic,
 * in-order processing of push and pull operations, running on a single Evaluator.
 * Partitions are based on the hash of the key.
 * Each partition consists of a queue, kvStore, and thread.
 *
 * This manager does NOT handle server or worker faults.
 */
@DriverSide
public final class PartitionedParameterServerManager implements ParameterServerManager {
  private final int numServers;
  private final int numPartitions;
  private final int queueSize;
  private final AtomicInteger workerCount;
  private final AtomicInteger serverCount;

  @Inject
  private PartitionedParameterServerManager(@Parameter(NumServers.class) final int numServers,
                                            @Parameter(NumPartitions.class) final int numPartitions,
                                            @Parameter(ServerQueueSize.class) final int queueSize) {
    this.numServers = numServers;
    this.numPartitions = numPartitions;
    this.queueSize = queueSize;
    this.workerCount = new AtomicInteger(0);
    this.serverCount = new AtomicInteger(0);
  }

  /**
   * Returns worker-side service configuration.
   * Sets {@link PartitionedParameterWorker} as the {@link ParameterWorker} class.
   */
  @Override
  public Configuration getWorkerServiceConfiguration() {
    final int workerIndex = workerCount.getAndIncrement();

    return Tang.Factory.getTang()
        .newConfigurationBuilder(ServiceConfiguration.CONF
            .set(ServiceConfiguration.SERVICES, PartitionedParameterWorker.class)
            .set(ServiceConfiguration.ON_CONTEXT_STOP, ContextStopHandler.class)
            .build())
        .bindImplementation(ParameterWorker.class, PartitionedParameterWorker.class)
        .bindImplementation(AsyncWorkerHandler.class, PartitionedWorkerHandler.class)
        .bindImplementation(ServerResolver.class, StaticServerResolver.class)
        .bindNamedParameter(NumServers.class, Integer.toString(numServers))
        .bindNamedParameter(NumPartitions.class, Integer.toString(numPartitions))
        .bindNamedParameter(EndpointId.class, WORKER_ID_PREFIX + workerIndex)
        .build();
  }

  /**
   * Returns server-side service configuration.
   */
  @Override
  public Configuration getServerServiceConfiguration() {
    final int serverIndex = serverCount.getAndIncrement();

    return Tang.Factory.getTang()
        .newConfigurationBuilder(ServiceConfiguration.CONF
            .set(ServiceConfiguration.SERVICES, PartitionedParameterServer.class)
            .build())
        .bindImplementation(PartitionedServerSideReplySender.class, PartitionedServerSideReplySenderImpl.class)
        .bindNamedParameter(EndpointId.class, SERVER_ID_PREFIX + serverIndex)
        .bindNamedParameter(PSMessageHandler.class, PartitionedServerSideMsgHandler.class)
        .bindImplementation(ServerResolver.class, StaticServerResolver.class)
        .bindNamedParameter(NumServers.class, Integer.toString(numServers))
        .bindNamedParameter(NumPartitions.class, Integer.toString(numPartitions))
        .bindNamedParameter(ServerQueueSize.class, Integer.toString(queueSize))
        .build();
  }
}
