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
package edu.snu.reef.dolphin.ps.ns;

import org.apache.reef.evaluator.context.events.ContextStart;
import org.apache.reef.evaluator.context.events.ContextStop;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.annotations.Unit;
import org.apache.reef.wake.EventHandler;
import org.apache.reef.wake.IdentifierFactory;

import javax.inject.Inject;
import java.util.logging.Logger;

/**
 * Register and unregister identifiers to and from a NetworkConnectionService
 * when contexts spawn/terminate, respectively.
 */
@Unit
public final class NetworkContextRegister {
  private static final Logger LOG = Logger.getLogger(NetworkContextRegister.class.getName());

  private final PSNetworkSetup psNetworkSetup;
  private final IdentifierFactory identifierFactory;
  private final String endpointId;

  @Inject
  private NetworkContextRegister(final PSNetworkSetup psNetworkSetup,
                                 final IdentifierFactory identifierFactory,
                                 @Parameter(EndpointId.class) final String endpointId) {
    this.psNetworkSetup = psNetworkSetup;
    this.identifierFactory = identifierFactory;
    this.endpointId = endpointId;
  }

  @Inject
  private NetworkContextRegister(final PSNetworkSetup psNetworkSetup,
                                 final IdentifierFactory identifierFactory) {
    this(psNetworkSetup, identifierFactory, null);
  }

  public final class RegisterContextHandler implements EventHandler<ContextStart> {
    @Override
    public void onNext(final ContextStart contextStart) {
      psNetworkSetup.registerConnectionFactory(identifierFactory.getNewInstance(
          endpointId == null ? contextStart.getId() : endpointId));
    }
  }

  public final class UnregisterContextHandler implements EventHandler<ContextStop> {
    @Override
    public void onNext(final ContextStop contextStop) {
      psNetworkSetup.unregisterConnectionFactory();
    }
  }
}
