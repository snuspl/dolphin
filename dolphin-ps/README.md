Parameter Server
================

The idea of using a separate server for aggregating and storing parameters for asynchronous machine learning algorithms has been proved to be very effective, in terms of learning speed as well as the accuracy of the output model ([DistBelief](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf), [Petuum](http://www.cs.cmu.edu/~./seunghak/petuum-13-weidai.pdf), [Adam](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-chilimbi.pdf)). One of the main assumptions of a parameter server application is that the given algorithm should be robust to inconsistencies between model replicas. Indeed, this is true for popular algorithms used nowadays. For example, several variants of [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) go through training sets without analyzing each and every training data sequentially. The fact that typical data sets consist of millions or billions of instances also contribute to such robustness of modern machine learning applications; in fact, inconsistency can be viewed as a method to avoid overfitting over training data.

### Architecture

<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Parameter-Server.png" alt="Parameter Server Architecture" width="484px" height="248px"/></p>

Dolphin's parameter server module is made up of two components: the server and the worker. Both sides have a message sender and a message handler (receiver) to communicate with each other using [Apache REEF](http://reef.incubator.apache.org/)'s Network Connection Service. The worker provides methods for applications to **push** and **pull** parameter values to/from servers, while the server receives such values and stores them in a key-value store. Before storing values, the server transforms them into a different format using a parameter updater in case the application requires a pre-processing step. [Adam](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-chilimbi.pdf), for instance, makes servers compute weight updates from activation and error gradient vectors rather than directly sending the updates from workers.

The code, as of Sep 2015, contains an implementation of a single-node parameter server (`SingleNodeParameterServer` and `SingleNodeParameterWorker`). However, the server does not necessarily need to be a single node; it can consist of more than one node, given the fact that workers send their parameters to the correct server node. We are currently working on a parameter server implementation that uses several nodes to process worker requests. One possible approach is to place a router node between workers and servers that directs worker requests to the corresponding server, similar to a Hadoop NameNode. Another approach is to configure workers to know which server owns which key, a problem that has been explored deeply in [distributed hash tables](https://en.wikipedia.org/wiki/Distributed_hash_table).

The worker provides the following APIs for applications:
* `push(key, preValue)`: send a value, associated with a key, to the server to be processed and stored
* `pull(key)`: fetch a value, which is associated with the given key, from the server

### How to use
 *The code examples below follow Apache REEF's coding style conventions; take a look at a few [REEF examples](https://cwiki.apache.org/confluence/display/REEF/Tutorials) if you're having a hard time understanding these examples.*
 
First, create [codec classes](https://github.com/apache/incubator-reef/blob/master/lang/java/reef-common/src/main/java/org/apache/reef/io/serialization/Codec.java) that specifies how Dolphin will encode and decode your keys, preValues, and values. In case you use classes that are Java Serializables, you can simply use [`SerializableCodec`](https://github.com/apache/incubator-reef/blob/master/lang/java/reef-common/src/main/java/org/apache/reef/io/serialization/SerializableCodec.java) although this is not recommended due to slow performance. You should also create a parameter updater class that shows how to process preValues and generate value updates.

Second, use `ParameterServerConfigurationBuilder` to build a Tang configuration of the parameter server module and submit it to REEF together with the REEF Driver configuration.
```Java
final Configuration psConf = ParameterServerConfigurationBuilder.newBuilder()
    .setManagerClass(SingleNodeParameterServer.class)
    .setUpdaterClass(MyCustomParameterUpdater.class)
    .setKeyCodecClass(MyKeyCodec.class)
    .setPreValueCodecClass(MyPreValueCodec.class)
    .setValueCodecClass(MyValueCodec.class)
    .build();

final Configuration finalDriverConf = Configurations.merge(driverConf, psConf);
driverLauncher.run(finalDriverConf);
```

Later at the driver, receive a Tang injection of `ParameterServerDriver` and call the `get*Configuration` methods for REEF contexts and services, both worker-side and server-side.
```Java
@Inject
private MyDriver(final ParameterServerDriver psDriver) {
  this.psDriver = psDriver;
}

final class ActiveContextHandler implements EventHandler<ActiveContext> {
  @Override
  public void onNext(final ActiveContext activeContext) {
  ...
  // when submitting a worker-side context and service
  final Configuration finalWorkerContextConf = Configurations.merge(
      workerContextConf, psDriver.getWorkerContextConfiguration());
  final Configuration finalWorkerServerConf = Configurations.merge(
      workerServiceConf, psDriver.getWorkerServiceConfiguration());
  activeContext.submitContextAndService(finalWorkerContextConf, finalWorkerServiceConf);

  // do the same for the server-side
  ...
  }
}
```

Finally, receive a Tang injection of a `ParameterWorker` instance at your application, and call its `push` and `pull` methods to communicate with the server.

```Java
@Inject
private MyTask(final ParameterWorker<MyKey, MyPreValue, MyValue> worker) {
  this.worker = worker;
}

@Override
public byte[] call(final byte[] bytes) throws Exception {
  ...
  worker.push(MY_KEY, MY_PREVALUE)
  final MyValue MY_VALUE = worker.pull(MY_KEY);
  ...
}

```
