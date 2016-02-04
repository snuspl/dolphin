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
package edu.snu.dolphin.dnn.layerparam.initializer;

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.conf.LayerConfigurationParameters;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;

import static edu.snu.dolphin.dnn.util.NeuralNetworkUtils.getShapeLength;
import static edu.snu.dolphin.dnn.util.NeuralNetworkUtils.shapeFromString;

/**
 * Parameter Initializer of fully connected layer.
 *
 * This class initializes the weight matrix
 * with pseudo random normal distributed value with mean 0 and given standard deviation.
 * This class initializes the bias vector with the given value.
 */
public final class FullyConnectedLayerParameterInitializer implements LayerParameterInitializer {

  private final MatrixFactory matrixFactory;
  private final int index;
  private final int numInput;
  private final int numOutput;
  private final float initWeight;
  private final float initBias;
  private final long randomSeed;

  @Inject
  public FullyConnectedLayerParameterInitializer(
      final MatrixFactory matrixFactory,
      @Parameter(LayerConfigurationParameters.LayerIndex.class) final int index,
      @Parameter(LayerConfigurationParameters.LayerInputShape.class) final String inputShape,
      @Parameter(LayerConfigurationParameters.NumberOfOutput.class) final int numOutput,
      @Parameter(LayerConfigurationParameters.RandomSeed.class) final long randomSeed,
      @Parameter(LayerConfigurationParameters.InitialWeight.class) final float initWeight,
      @Parameter(LayerConfigurationParameters.InitialBias.class) final float initBias) {
    this.matrixFactory = matrixFactory;
    this.index = index;
    this.randomSeed = randomSeed;
    this.numInput = getShapeLength(shapeFromString(inputShape));
    this.numOutput = numOutput;
    this.initWeight = initWeight;
    this.initBias = initBias;
  }

  /** {@inheritDoc} */
  @Override
  public LayerParameter generateInitialParameter() {
    final Matrix weight = matrixFactory.randn(numOutput, numInput, randomSeed);
    final Matrix bias = matrixFactory.create(numOutput).fill(initBias);

    weight.muli(initWeight); // multiply by standard deviation.

    return LayerParameter.newBuilder()
        .setWeightParam(weight)
        .setBiasParam(bias)
        .build();
  }

  /** {@inheritDoc} */
  @Override
  public int getIndex() {
    return this.index;
  }

  @Override
  public int[] getOutputShape() {
    return new int[]{numOutput};
  }
}
