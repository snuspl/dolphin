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

import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.LayerIndex;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.ActivationFunction;
import edu.snu.reef.dolphin.neuralnet.conf.LayerConfigurationParameters.NumberOfOutput;
import edu.snu.reef.dolphin.neuralnet.layerparam.initializer.LayerParameterInitializer;
import org.apache.reef.tang.annotations.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.inject.Inject;

/**
 * Fully connected Layer.
 */
public final class FullyConnectedLayer extends LayerBase {

  private final String activationFunction;

  @Inject
  public FullyConnectedLayer(@Parameter(LayerIndex.class) final int index,
                             @Parameter(ActivationFunction.class) final String activationFunction,
                             @Parameter(NumberOfOutput.class) final int numOutput,
                             final LayerParameterInitializer layerParameterInitializer) {
    super(index, numOutput);
    this.activationFunction = activationFunction;
    setLayerParameter(layerParameterInitializer.generateInitialParameter());
  }

  /** {@inheritDoc} */
  @Override
  public boolean isLearnable() {
    return true;
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
  public INDArray backPropagate(final INDArray activation,
                                final LayerParameter nextParameter,
                                final INDArray nextError) {
    return nextError.mmul(nextParameter.getWeightParam().transpose()).muli(derivative(activation));
  }

  /** {@inheritDoc} */
  @Override
  public INDArray backPropagate(final INDArray activation, final INDArray label) {
    return activation.sub(label);
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateParameterGradient(final INDArray input, final INDArray error) {
    return LayerParameter.newBuilder()
        .setWeightParam(input.transpose().mmul(error))
        .setBiasParam(error)
        .build();
  }
}
