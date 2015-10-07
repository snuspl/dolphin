Dolphin
=======

Dolphin is a machine learning platform built on top of [Apache REEF](http://reef.incubator.apache.org/). Dolphin consists of a BSP-style machine learning framework (`dolphin-bsp`), a deep learning framework (`dolphin-dnn`), and a parameter server module (`dolphin-ps`).

### Submodules

* [`dolphin-bsp`](dolphin-bsp/README.md): BSP-style framework for multi-staged processing.
* [`dolphin-dnn`](dolphin-dnn/README.md): Deep learning framework for training deep neural network models on large datasets.
* [`dolphin-ps`](dolphin-ps/README.md): Parameter server module for asynchronous machine learning algorithms.

### How to build and run Dolphin
1. Build REEF: check https://cwiki.apache.org/confluence/display/REEF/Compiling+REEF  
  Currently, Dolphin depends on REEF `0.13.0-incubating-SNAPSHOT`, which means you must build the current snapshot of REEF before building Dolphin. We will move to the release version `0.13.0-incubating` once it's out, around Oct 2015.

2. Build Dolphin:
    ```
    git clone https://github.com/cmssnu/dolphin
    cd dolphin
    mvn clean install
    ```
    

3. Run Dolphin: check the READMEs in the submodules for more details.

### Dolphin Mailing List
Bug reports and feature requests are welcome, as well as even simple questions!  
Contact us and share your thoughts by subscribing to dolphin-discussion@googlegroups.com.

