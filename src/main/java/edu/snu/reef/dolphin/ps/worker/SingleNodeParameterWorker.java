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
package edu.snu.reef.dolphin.ps.worker;

import edu.snu.reef.dolphin.ps.driver.SingleNodeParameterServerManager;
import org.apache.reef.annotations.audience.EvaluatorSide;
import org.apache.reef.tang.InjectionFuture;
import org.apache.reef.tang.annotations.Parameter;

import javax.annotation.concurrent.ThreadSafe;
import javax.inject.Inject;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

/**
 * Parameter Server worker that interacts with a server which uses only one node.
 * A single instance of this class can be used by more than one thread safely.
 */
@EvaluatorSide
@ThreadSafe
public final class SingleNodeParameterWorker<K, P, V> implements ParameterWorker<K, P, V> {
  private static final Logger LOG = Logger.getLogger(SingleNodeParameterWorker.class.getName());
  private static final long TIMEOUT = 10000; // milliseconds

  /**
   * Network Connection Service identifier of the server.
   */
  private final String serverId;

  /**
   * Send messages to the server using this field.
   * Without {@link InjectionFuture}, this class creates an injection loop with
   * classes related to Network Connection Service and makes the job crash (detected by Tang).
   */
  private final InjectionFuture<WorkerSideMsgSender<K, P>> sender;

  /**
   * Map for caching server replies so that this class can provide values to multiple requests of the same key.
   */
  private final ConcurrentMap<K, ValueWrapper> keyToValueWrapper;

  @Inject
  private SingleNodeParameterWorker(@Parameter(SingleNodeParameterServerManager.ServerId.class) final String serverId,
                                    final InjectionFuture<WorkerSideMsgSender<K, P>> sender) {
    this.serverId = serverId;
    this.sender = sender;
    this.keyToValueWrapper = new ConcurrentHashMap<>();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void push(final K key, final P preValue) {
    sender.get().sendPushMsg(serverId, key, preValue);
  }

  /**
   * Try to fetch a {@code value} from the server, waiting for a certain {@code TIMEOUT} period.
   * @param key key object representing the expected value
   * @return value specified by the {@code key}, or null if wait time exceeds {@code TIMEOUT}
   */
  @Override
  public V pull(final K key) {
    while (true) {
      final boolean isFirstToWait = keyToValueWrapper.putIfAbsent(key, new ValueWrapper()) == null;

      final ValueWrapper valueWrapper = keyToValueWrapper.get(key);
      synchronized (valueWrapper) {
        // the reply arrived right before I acquired the lock
        // try again from start
        if (!keyToValueWrapper.containsKey(key)) {
          continue;
        }

        // the first one to wait will send the fetch message
        // others don't have to send the same message again
        if (isFirstToWait) {
          sender.get().sendPullMsg(serverId, key);
        }

        try {
          valueWrapper.wait(TIMEOUT);
        } catch (final InterruptedException ex) {
          throw new RuntimeException("InterruptedException while waiting for reply for key " + key);
        }

        return valueWrapper.getValue();
      }
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void reply(final K key, final V value) {
    final ValueWrapper valueWrapper = keyToValueWrapper.remove(key);
    synchronized (valueWrapper) {
      valueWrapper.setValue(value);
      // wake all threads waiting for the value
      valueWrapper.notifyAll();
    }
  }

  /**
   * Wrapper class needed to distinguish null from non-null objects without causing {@code NullPointerException}s.
   */
  private final class ValueWrapper {
    private V value;

    public void setValue(final V value) {
      this.value = value;
    }

    public V getValue() {
      return this.value;
    }
  }
}
