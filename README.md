Dolphin
=======

Dolphin is a machine learning platform built top on [Apache REEF](http://reef.incubator.apache.org/). Dolphin contains a BSP-style framework (`dolphin-bsp`), a deep learning framework (`dolphin-dnn`), and a parameter server module (`dolphin-ps`).

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
We appreciate bug reports, feature requests, or even simple questions!  
Subscribe to dolphin-discussion@googlegroups.com and we will reply ASAP.

