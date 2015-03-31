package edu.snu.reef.flexion.core;

import edu.snu.reef.flexion.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.flexion.groupcomm.interfaces.DataGatherSender;
import edu.snu.reef.flexion.groupcomm.interfaces.DataReduceSender;
import edu.snu.reef.flexion.groupcomm.interfaces.DataScatterReceiver;

/**
 * Abstract class for user-defined compute tasks.
 * This class should be extended by user-defined compute tasks that override run, initialize, and cleanup methods
 */
public abstract class UserComputeTask <T> {

    /**
     * Main process of a user-defined compute task
     * @param iteration
     * @param data
     */
    public abstract void run(int iteration, T data);

    /**
     * Initialize a user-defined compute task.
     * Results of the previous task can be retrieved from the given key-value store
     * @param keyValueStore
     */
    public void initialize(KeyValueStore keyValueStore) {
    }

    /**
     * Clean up a user-defined compute task
     * Results of the current task can be passed to the next task through the given key-value store
     * @param keyValueStore
     */
    public void cleanup(KeyValueStore keyValueStore) {
    }

    final public boolean isReduceUsed() {
       return (this instanceof DataReduceSender);
    }

    final public boolean isGatherUsed() {
        return (this instanceof DataGatherSender);
    }

    final public boolean isBroadcastUsed() {
        return (this instanceof DataBroadcastReceiver);
    }

    final public boolean isScatterUsed() {
        return (this instanceof DataScatterReceiver);
    }
}
