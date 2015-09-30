# Dolphin - Deep Neural Network

Dolphin’s deep neural network is a deep learning framework on the top of [Apache REEF](https://reef.incubator.apache.org). It supports a BSP-style deep learning and an asynchronous deep learning by communicating with a parameter server. It is designed for training big deep learning models over large datasets by supporting both data partitioning and model partitioning inspired by Google's [DistBelief](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf). Model partitioning is on-going work.

* Data partitioning: All input data are distributed to evaluators each of which has a replica of a whole neural network model and processes training against a part of data.

![Data Partitioning](http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Data-Partitioning.png)

* Model partitioning: Each evaluator processes training of a part of a whole neural network model.

![Model Partitioning](http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Model-Partitioning.png)

Currently, Dolphin's deep neural network only provides fully connected layer. More types of layers such as convolutional layer and subsampling layer will be supported.

## Architecture

![Dolphin DNN Architecture](http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/DNN-Architecture.png)

Dolphin's deep neural network is made up of two components; a neural network model and a parameter provider. A neural network model consists of layers that are defined by a [protocol buffer definition file](#layers). A parameter provider is an instance that receives parameter gradients from a neural network model and sends these gradients to a parameter server that generates updated parameters using received gradients.

The training procedure of a neural network model is following. First, each evaluator builds its neural network model by a neural network configuration  that is provided by REEF driver. After being built, the neural network model replica computes activation values for each layer with given training input data. Using these activation values, it computes parameter gradients for each layer by backpropagation and pushes these gradients to a parameter provider. The parameter provider sends gradients to a parameter server. The parameter server updates parameters of the model using gradients received from neural network replicas. Each neural network replica requests and receives updated parameters from the parameter server, and replaces their parameters with updated ones periodically.

## Input file format
Dolphin's deep neural network can process a Numpy compatible plain text file which is stored in the following format. 

![Input File Format](http://cmslab.snu.ac.kr/home/wp-content/uploads/2015/09/Input-Data-Format.png)

Each line represents a vector whose elements are separated using a delimiter specified by a command line parameter [`delim`](#parameter-delim). This vector consists of a serialized input data and its metadata. We assume that each element can be converted to a floating number.

* *Serialized input data*: an input data that is serialized as a vector. The shape of an input data can be specified in a [protocol buffer definition file](#configuration-input_shape).
* *Output*: an expected output for an input data.
* *Validation flag*: a flag that indicates whether an input data is used for validation. `1.0` if a data is used for validating a model. `0.0` if a data is used for training a model.

## Configuration
To create a neural network model, you need to define the architecture of a neural network model in a protocol buffer definition file.
Now, only fully connected layer is supported. 

### Common
* Required
	* `batch_size`: the number of training inputs that is used for each parameters update.
	* `stepsize`: a step size (learning rate) for stochastic gradient descent.
	* <a name=configuration-input_shape>`input_shape`</a>: the shape of input data.

### Parameter Provider
Parameter provider is an instance that receives parameter gradients for each training input from a neural network model and provides the model with updated parameters (weights and biases).

### Local Parameter Provider
A local parameter provider does not communicate with a parameter server. Instead, it computes updated parameters using gradients received from a neural network model by itself and provides updated parameters on a request of the model.

* Parameter provider type: `Local`

### Group Communication Parameter Provider
A group communication provider is used for a BSP-style deep learning. The group communication provider communicates with a group communication parameter server using [Apache REEF](https://reef.incubator.apache.org)'s Group Communication Service. The group communication parameter server aggregates parameter gradients received from each neural network replica by reduce operation. It updates parameters of a model using gradients and broadcasts updated parameters to all neural network replicas.

* Parameter provider type: `GroupComm`

### Parameter Server Parameter Provider
A parameter server parameter provider is used for an asynchronous deep learning. The parameter server parameter provider sends parameter gradients that its model replica computes to Dolphin's parameter server. Dolphin's parameter server updates parameters of a model by a specified batch size and provides a replica with latest updated parameters when it receives a request for updated parameters. In contrast to a group communication parameter provider, after replacing its parameters with the updated one, each replica proceeds training for next input data without waiting for other replicas updates. Thus, there is inconsistency that parameters used for training of a replica can be different from each other.

* Parameter provider type: `ParameterServer`

### Layers

#### Fully Connected Layer
* Layer type: `FullyConnected`
* Parameters(`FullyConnectedLayerConfiguration fully_connected_param`)
	* Required
		* `init_weight`: a standard deviation that is used to initialize the weights in this layer from a Gaussian distribution with mean 0.
		* `init_bias`: a constant value with which the biases in this layer are initialized.
		* `activation_function`: an activation function to produce a output value for this layer.

**More types of layers such as convolutional layer and subsampling layer will be supported.**

#### Activation function
The following functions are supported.

* Sigmoid: `sigmoid`
* ReLU: `relu`
* TanhH: `tanh`
* Power: `pow` (Now, this produces squared value.)
* Absolute: `abs`
* Softmax: `softmax`

## How to run
The script for running a neural network model is located at `bin/run_neuralnetwork.sh`. `test/resources/data/neuralnet` is a sample of [MNIST](http://yann.lecun.com/exdb/mnist) dataset composed of 1,000 training images and 100 test images. `test/resources/configuration/neuralnet` is an example of a protocol buffer definition file and defines a neural network model that has two fully connected layers with a local parameter provider.

You can run training of the model example with the sample of MNIST dataset on local runtime environment by

```bash
cd $DOLPHIN_HOME
bin/run_neuralnet.sh -local true -maxIter 100 -conf src/test/resources/configuration/neuralnet -input src/test/resources/data/neuralnet -timeout 800000
```

You can specify your neural network training environment with the following command line parameters.

Command line parameters

* Required
	* `input`: the path of an input data file.
	* `conf`: the path of a protocol buffer definition file.
* Optional
	* `local`[default=false]: the flag that indicates a local runtime environment. If `false`, neural network will run on YARN environment.
	* `maxIter`[default=20]: the maximum number of allowed iterations before a neural network training stops.
	* <a name="parameter-delim">`delim`</a>\[default=,\]: the delimiter that is used for separating elements of input data.
	* `timeout`[default=100000]: an allowed time until neural network training ends. (unit: millisecond)

## Example
### A example of protocol buffer definition file for MNIST
The following is the example of a protocol buffer definition file for MNIST `test/resources/configuration/neuralnet`. 

```protobuf
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
A neural network model comprises two fully connected layers with 50 and 10 features respectively with a local parameter provider. `input_shape` specifies the shape of input data. For MNIST dataset, each data is 28 * 28 images. So, `input_shape` is configured as following.

```protobuf
input_shape {
  dim: 28
  dim: 28
}
```
Parameters are updated with gradients when every 10 inputs are processed since `batch_size` is specified as 10 and 1e-3 is used as learning rate for stochastic gradient descent algorithm.