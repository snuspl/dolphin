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
package edu.snu.reef.dolphin.neuralnet.layers;

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Fully connected Layer.
 */
public final class FullyConnectedLayer implements Layer {

  protected final int index;
  private final String activationFunction;
  private final int numOutput;
  private LayerParameter layerParameter;

  @Inject
  public FullyConnectedLayer(@Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
                             @Parameter(LayerConfigurationParameters.ActivationFunction.class)
                                 final String activationFunction,
                             @Parameter(LayerConfigurationParameters.NumberOfOutput.class) final int numOutput,
                             final LayerParameterInitializer layerParameterInitializer) {
    this.index = index;
    this.activationFunction = activationFunction;
    this.numOutput = numOutput;
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }


  /** {@inheritDoc} */
  @Override
  public int getIndex() {
    return index;
  }

  /** {@inheritDoc} */
  @Override
  public int getNumOutput() {
    return numOutput;
  }

  /** {@inheritDoc} */
  @Override
  public void setLayerParameter(final LayerParameter layerParameter) {
    this.layerParameter = layerParameter;
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter getLayerParameter() {
    return this.layerParameter;
  }

  /** {@inheritDoc} */
  @Override
  public INDArray derivative(final INDArray activation) {
    return Nd4j.getExecutioner().execAndReturn(
        Nd4j.getOpFactory().createTransform(activationFunction, activation.dup()).derivative());
  }

  /** {@inheritDoc} */
  @Override
  public INDArray feedForward(final INDArray input) {
    final INDArray output =
        input.mmul(getLayerParameter().getWeightParam()).addiRowVector(getLayerParameter().getBiasParam());
    return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, output));
  }

  /** {@inheritDoc} */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray derivative,
                                final LayerParameter prevParam, final INDArray nextGradient) {
    return nextGradient.mmul(prevParam.getWeightParam().transpose()).muli(derivative);
  }

  /** {@inheritDoc} */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray label) {
    return activation.sub(label);
  }
}
