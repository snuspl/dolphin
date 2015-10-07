Dolphin - Bulk Synchronous Parallel (BSP) Framework
===================================================

Many large-scale machine learning applications and frameworks run algorithms under distributed environments by dividing the whole computation into several independent partitions and aggregating the results of each partition every `n` iterations. This aggregating process can be viewed as a synchronization step, where workers synchronize with each other to get ready for the next iteration. The [Bulk Synchronous Parallel (BSP)](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel) model is one of the most well-known programming models to suggest such synchronous behavior, and although recent researches argue asynchronous models to be more fast and accurate, synchronous models are still being widely used to solve modern machine learning problems.

### Architecture
<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Dolphin-BSP.png" alt="Dolphin BSP Architecture" width="483px" height="215px"/></p>

The BSP module of Dolphin is a machine learning framework for running ML algorithms, BSP style. `dolphin-bsp` creates several Compute Tasks for distributed computation, and a single Controller Task for aggregating the computation results sent from the Compute Tasks. The Controller Task starts a job by sending Compute Tasks the initial model to work on. Compute Tasks perform local computation using the given model, and then transmits their calculations back to the Controller Task. After aggregation is done, the Controller Task starts another iteration by sending an updated model to Compute Tasks, and the same process repeats over and over until the model converges. Communication between Tasks is done using [Apache REEF](http://reef.incubator.apache.org)’s Group Communication Service, which provides MPI Broadcast, Reduce, Scatter, and Gather operations.

An algorithm may need more than one Controller-Compute **stage** to completely run; for example, a typical application loads data from a backing distributed file system and pre-processes it before actually running the core computation. Each stage can be regarded as its own BSP stage having its own set of Tasks. Dolphin supports such multiple BSP stages by running a separate set of Controller Task and Compute Tasks per stage. The overhead of spawning a different set of Tasks for each stage is minimal thanks to the retainability of REEF Evaluators throughout the whole job. An Evaluator that was used as a container for a Compute Task of a certain stage is re-used for a Compute Task of the next stage (same for the Controller Task).


### How to write a new algorithm
 *The code examples below follow Apache REEF's coding style conventions; take a look at a few [REEF examples](https://cwiki.apache.org/confluence/display/REEF/Tutorials) if you're having a hard time understanding these examples.*

Here, we describe how you can write your own algorithm on Dolphin. Examples are provided in the [`edu.snu.dolphin.bsp.examples.ml.algorithms`](https://github.com/cmssnu/dolphin/tree/master/dolphin-bsp/src/main/java/edu/snu/dolphin/bsp/examples/ml/algorithms) package. There are roughly four steps:

First, create classes specifying what kind of local computation the Controller Task and Compute Tasks should do. You must also pick what kind of MPI operations you will use by making your class implement the corresponding interfaces. If your algorithm requires more than one stage, then you should write a Controller/Compute Task for each stage.
```Java
public class MyFirstCtrlTask extends UserControllerTask implements DataBroadcastSender<Integer>, DataReduceReceiver<Integer> {
  @Override
  public void run(final int iteration) { ... }

  @Override
  public void receiveReduceData(final int iteration, final Integer data) { ... }

  @Override
  public Integer sendBroadcastData(final int iteration) { ... }
}
```

Next, define your algorithm's stages as a `UserJobInfo`. Codec classes that indicate how your data will be serialized into bytes must be input in this step too, as well as a Reduce function class in case you use MPI Reduce, and your data parser class.
```Java
public class MyJobInfo implement UserJobInfo {

  @Inject
  private MyJobInfo() { }

  @Override
  public List<StageInfo> getStageInfoList() {
    final List<StageInfo> stageInfoList = new ArrayList<>(1);

    stageInfoList.add(
        StageInfo.newBuilder(MyFirstCmpTask.class, MyFirstCtrlTask.class, CommunicationGroup.class)
            .setBroadcast(MyBroadcastCodec.class)
            .setReduce(MyReduceCodec.class, MyReduceFunction.class)
            .build());

    return stageInfoList;
  }

  @Override
  public Class<? extends DataParser> getDataParser() { return MyDataParser.class; }
}
```

Third, create a `UserParameters` class that contains the algorithm parameters and configurations you will use.
```Java
public class MyParameters implements UserParameters {
  ...
  @Override
  public Configuration getUserCmpTaskConf() {
    return Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(StepSize.class, "10")
        .bindNamedParameter(Lambda.class, "0.9")
        .bindImplementation(Loss.class, LogisticLoss.class)
        .build();
  }
  ...
}
```


Last, use `DolphinLauncher` to run your algorithm, using the classes you created up until now.
```Java
DolphinLauncher.run(
    Configurations.merge(
        DolphinConfiguration.getConfiguration(args, MyParameters.getCommandLine()),
        Tang.Factory.getTang().newConfigurationBuilder()
            .bindNamedParameter(JobIdentifier.class, "My Algorithm")
            .bindImplementation(UserJobInfo.class, MyJobInfo.class)
            .bindImplementation(UserParameters.class, MyParameters.class)
            .build()
      )
```
