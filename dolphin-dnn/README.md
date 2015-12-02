# Dolphin - Deep Neural Network

`dolphin-dnn` is a deep learning framework built on [Apache REEF](https://reef.apache.org). It is capable of both BSP-style synchronous deep learning and parameter server-backed asynchronous deep learning. `dolphin-dnn` is designed for training large neural network models on big data by supporting data partitioning as well as model partitioning, inspired by Google's [DistBelief](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf), although the current codebase only contains methods for data partitioning; model partitioning is on-going work.

* Data partitioning: Input data are distributed across evaluators, each of which has a replica of the whole neural network model. Every replica independently trains its model on its own data, and the updated models are shared between replicas periodically. The model sharing can be done either synchronously or asynchronously, depending on the implementation.

<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Data-Partitioning.png" alt="Data Partitioning" width="646px" height="410px"/></p>

* Model partitioning: Each partition works on a certain portion of the neural network model. Partitions of a model need to process the same training data at a given time, whereas in data partitioning model replicas make progress without regard to each other.

<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Model-Partitioning.png" alt="Model Partitioning" width="646px" height="374px"/></p>

Currently `dolphin-dnn` only supports fully connected layers, but other types of layers such as convolutional layers and subsampling layers will be supported in the future.

## Architecture

<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/DNN-Architecture.png" alt="Dolphin DNN Architecture" width="646px" height="364px"/></p>

A typical REEF evaluator in `dolphin-dnn` is made up of two components; a neural network model and a parameter provider. A neural network model consists of layers that are defined by a [protocol buffer definition file](#layers). A parameter provider is an instance that receives parameter gradients from the neural network model and sends these gradients to a parameter server, which in turn generates new parameters using gradients.

The training procedure of a neural network model is as follows. First, each REEF evaluator builds its neural network model from a configuration that is provided by the REEF driver. This neural network model replica then computes activation values for each layer with given training input data. Using these activation values, the model computes parameter gradients for each layer via backpropagation and hands these gradients to the parameter provider. The parameter provider acts as a communication media between the model and the server, by interacting with the server to update model parameters and providing new parameters for the local model. The implementation of a parameter provider and server may differ per design (see 'Parameter Provider' section below). After receiving the new parameters, each model replica repeats the above steps with the update parameters for many epochs.

## Input file format
`dolphin-dnn` can process Numpy-compatible plain text input files which are stored in the following format.

<p align="center"><img src="http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Input-Data-Format.png" alt="Input File Format" width="625px" height="202px"/></p>

Each line represents a vector whose elements are separated using a delimiter, specified via the command line parameter [`delim`](#parameter-delim). Vectors should consist of serialized input data and other metadata. We assume that each element can be converted to a floating number `float`.

* *Serialized input data*: an input data object that is serialized as a vector. The shape of input data can be specified in a separate [protocol buffer definition file](#configuration-input_shape).
* *Output*: the expected output for a given input data object.
* *Validation flag*: a flag that indicates whether an input data is used for validation: `1.0` for validating, and `0.0` for training.

## Configuration
To create a neural network model, you must define the architecture of your neural network model in a protocol buffer definition file. Only fully connected layers are supported, for now.

### Common Fields
* `batch_size`: the number of training inputs used per parameter update.
* `stepsize`: step size (learning rate) for stochastic gradient descent.
* <a name=configuration-input_shape>`input_shape`</a>: the shape of input data.

### Parameter Provider
The parameter provider is an instance that receives parameter gradients for each training input from a neural network model and provides the model with updated parameters (weights and biases). You must select the type of parameter provider you want to use by specifying the field `parameter_provider`.

##### Local Parameter Provider
Local parameter providers do not communicate with a separate parameter server. Instead, it locally updates parameters using gradients received from a neural network model. This provider is used mainly for testing the correctness of a network.
```
parameter_provider {
  type: "local"
}
```

##### Group Communication Parameter Provider
Group communication providers are used for BSP-style network training. The group communication provider communicates with a group communication parameter server using [Apache REEF](https://reef.apache.org)'s Group Communication Service. The server aggregates parameter gradients received from providers using the MPI Reduce operation. After updating parameters, the server broadcasts the updated parameters back to all providers. All operations are done synchronously, hence the name group communication.

```
parameter_provider {
  type: "groupcomm"
}
```

##### Parameter Server Parameter Provider
Parameter server parameter providers are used for asynchronous training. This provider is used together with Dolphin's parameter server module [`dolphin-ps`](../dolphin-ps/README.md). Parameter server providers can send parameter **push** or **pull** requests to the server. The server updates parameters of a model when gradients are **push**ed, and provides a model replica with the latest parameters when it receives a **pull** request. After replacing its parameters with the updated ones, each model replica proceeds with its next input data without waiting for other replicas to finish their updates, in contrast to the group communication parameter provider where all providers start with the same model weights. Thus, there is some inconsistency between replicas; the parameters used for training can be different from each other.


```
parameter_provider {
  type: "paramserver"
}
```

### Layers

##### Fully Connected Layer
* Layer type: `FullyConnected`
* Parameters (`FullyConnectedLayerConfiguration fully_connected_param`)
	* `init_weight`: the standard deviation that is used to initialize the weights in this layer from a Gaussian distribution with mean 0.
	* `init_bias`: constant value with which the biases of this layer are initialized.
	* `activation_function`: the activation function to produce a output value for this layer.

**More types of layers such as convolutional layer and subsampling layer will be supported.**

##### Activation function
The following functions are supported.

* Sigmoid: `sigmoid`
* ReLU: `relu`
* TanhH: `tanh`
* Power: `pow` (squared value)
* Absolute: `abs`
* Softmax: `softmax`

## How to run
A script for training a neural network model is included with the source code, in `bin/run_neuralnetwork.sh`. `test/resources/data/neuralnet` is a sample subset of the [MNIST](http://yann.lecun.com/exdb/mnist) dataset, composed of 1,000 training images and 100 test images. `test/resources/configuration/neuralnet` is an example of a protocol buffer definition file; it defines a neural network model that uses two fully connected layers and a local parameter provider.

You can run a network of the given example on REEF local runtime environment by

```bash
cd $DOLPHIN_HOME
bin/run_neuralnet.sh -local true -maxIter 100 -conf dolphin-dnn/src/test/resources/configuration/neuralnet -input dolphin-dnn/src/test/resources/data/neuralnet -timeout 800000
```

#### Command line parameters

* Required
	* `input`: path of the input data file to use.
	* `conf`: path of the protocol buffer definition file to use.
* Optional
	* `local`[default=false]: a boolean value that indicates whether to use REEF local runtime environment or not. If `false`, the neural network will run on YARN environment.
	* `maxIter`[default=20]: the maximum number of allowed iterations before the neural network training stops.
	* <a name="parameter-delim">`delim`</a>\[default=,\]: the delimiter that is used for separating elements of input data.
	* `timeout`[default=100000]: allowed time until neural network training ends. (unit: milliseconds)

## Example
### A example of protocol buffer definition file for MNIST
The following is the example of a protocol buffer definition file for the MNIST dataset. It can be found at `dolphin-dnn/src/test/resources/configuration/neuralnet`.

```
batch_size: 10
stepsize: 1e-3
input_shape {
  dim: 28
  dim: 28
}
parameter_provider {
  type: "local"
}
layer {
  type: "FullyConnected"
  num_input: 784
  num_output: 50
  fully_connected_param {
    init_weight: 1e-4
    init_bias: 2e-4
    activation_function: "sigmoid"
  }
}
layer {
  type: "FullyConnected"
  num_input: 50
  num_output: 10
  fully_connected_param {
    init_weight: 1e-2
    init_bias: 2e-2
    activation_function: "sigmoid"
  }
}
```
This model comprises two fully connected layers with 50 and 10 features, respectively, and a local parameter provider. `input_shape` specifies the shape of input data. For the MNIST dataset, each data object is a 28 * 28 image and thus `input_shape` is configured as the following.

```
input_shape {
  dim: 28
  dim: 28
}
```
Parameters are updated when every 10 inputs are processed since `batch_size` is specified as 10, and 1e-3 is used as the learning rate for the stochastic gradient descent algorithm.
