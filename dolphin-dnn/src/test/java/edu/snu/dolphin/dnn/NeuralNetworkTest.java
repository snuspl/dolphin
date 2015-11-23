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
package edu.snu.dolphin.dnn;

import edu.snu.dolphin.dnn.conf.ActivationLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.ActivationWithLossLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.FullyConnectedLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.layerparam.provider.LocalNeuralNetParameterProvider;
import edu.snu.dolphin.dnn.util.Nd4jUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Unit tests for neural network.
 */
public class NeuralNetworkTest {

  private final float tolerance = 1e-6f;

  private final INDArray input = Nd4j.create(new float[]{77, 57, 30, 26, 75, 74, 87, 75});
  private final INDArray expectedOutput =
      Nd4j.create(new float[]{5.96001904648e-01f, 5.54388443935e-01f, 4.88766051729e-01f});
  private final INDArray label = Nd4j.create(new float[]{0, 1, 0});
  private final int numHiddenUnits = 5;

  private final INDArray[] expectedActivations = new INDArray[] {
      Nd4j.create(new float[]{
          1.37635572407e-02f, 2.03896454414e-02f, 8.84165997677e-03f, 2.93623838992e-02f, -7.07258002822e-03f}),
      Nd4j.create(new float[]{
          5.03440834992e-01f, 5.05097234769e-01f, 5.02210400594e-01f, 5.07340068629e-01f, 4.98231862363e-01f}),
      Nd4j.create(new float[]{
          3.88833699304e-01f, 2.18417981391e-01f, -4.49433566656e-02f}),
      expectedOutput};

  private final Configuration neuralNetworkConfiguration = NeuralNetworkConfigurationBuilder.newConfigurationBuilder()
      .setBatchSize(1)
      .setStepsize(1e-2f)
      .setParameterProviderClass(LocalNeuralNetParameterProvider.class)
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumInput(input.length())
              .setNumOutput(numHiddenUnits)
              .setInitWeight(0.0001f)
              .setInitBias(0.0002f)
              .setRandomSeed(10)
              .build())
      .addLayerConfiguration(
          ActivationLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumInput(numHiddenUnits)
              .setNumOutput(numHiddenUnits)
              .setActivationFunction("sigmoid")
              .build())
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumInput(numHiddenUnits)
              .setNumOutput(expectedOutput.length())
              .setInitWeight(0.2f)
              .setInitBias(0.3f)
              .setRandomSeed(10)
              .build())
      .addLayerConfiguration(ActivationWithLossLayerConfigurationBuilder.newConfigurationBuilder()
          .setNumInput(expectedOutput.length())
          .setNumOutput(expectedOutput.length())
          .setActivationFunction("sigmoid")
          .setLossFunction("cross-entropy")
          .build())
      .build();

  private NeuralNetwork neuralNetwork;

  private final INDArray[] expectedErrors = new INDArray[] {
      Nd4j.create(new float[]{
          -1.10814514935e-02f, 4.75458113254e-02f, 2.79511566851e-02f, -3.76325218465e-02f, -6.66430042946e-02f}),
      Nd4j.create(new float[]{
          -4.43279052273e-02f, 1.90203012570e-01f, 1.11806811835e-01f, -1.50562534580e-01f, -2.66575350768e-01f}),
      Nd4j.create(new float[]{5.96001904648e-01f, -4.45611556065e-01f, 4.88766051729e-01f})};

  private final LayerParameter[] expectedParams = new LayerParameter[]{
      LayerParameter.newBuilder()
          .setWeightParam(Nd4j.create(new float[]{
              8.43271085004e-03f, 6.30096936863e-03f, 3.39385184743e-03f, 2.77477924060e-03f, 8.53602946125e-03f,
              8.44699779920e-03f, 9.53429478040e-03f, 8.27071991212e-03f, -3.66582312238e-02f, -2.70532006510e-02f,
              -1.43677246544e-02f, -1.24133925477e-02f, -3.55010755578e-02f, -3.51333945502e-02f, -4.12985268766e-02f,
              -3.56029583595e-02f, -2.14895112802e-02f, -1.59334007414e-02f, -8.38015241706e-03f, -7.31539494329e-03f,
              -2.08851161497e-02f, -2.06472822008e-02f, -2.44371878001e-02f, -2.08418701603e-02f, 2.91009853375e-02f,
              2.15053585334e-02f, 1.12523984594e-02f, 9.75559048423e-03f, 2.83249945339e-02f, 2.78617690825e-02f,
              3.28448331050e-02f, 2.82338715978e-02f, 5.12505138331e-02f, 3.79973748138e-02f, 1.96798049550e-02f,
              1.72828290869e-02f, 5.00290409316e-02f, 4.93010379396e-02f, 5.80023241160e-02f, 5.00251904698e-02f})
              .reshape(input.length(), numHiddenUnits))
          .setBiasParam(Nd4j.create(new float[]{
              3.10814514935e-04f, -2.75458113254e-04f, -7.95115668513e-05f, 5.76325218465e-04f, 8.66430042946e-04f})
              .reshape(1, numHiddenUnits))
          .build(),
      LayerParameter.EMPTY, // sigmoid activation layer
      LayerParameter.newBuilder()
          .setWeightParam(Nd4j.create(new float[]{
              -2.03014116811e-01f, 2.44876642191e-01f, 9.28304253913e-02f, 1.87009753641e-02f, 7.41970556867e-03f,
              -9.36696160145e-02f, -1.26948175865e-01f, -2.44954097078e-04f, 1.41093564760e-01f, -7.24960104289e-02f,
              6.32980868345e-02f, -3.33847090889e-02f, 1.07187527879e-01f, -2.10442219526e-01f, -6.28627854949e-01f})
              .reshape(numHiddenUnits, expectedOutput.length()))
          .setBiasParam(Nd4j.create(new float[]{2.94039980954e-01f, 3.04456115561e-01f, 2.95112339483e-01f})
              .reshape(1, expectedOutput.length()))
          .build(),
      LayerParameter.EMPTY}; // sigmoid activation layer

  @Before
  public void buildNeuralNetwork() throws InjectionException {
    final Injector injector = Tang.Factory.getTang().newInjector(neuralNetworkConfiguration);
    neuralNetwork = injector.getInstance(NeuralNetwork.class);
  }

  /**
   * Unit test for feedforward of neural network.
   */
  @Test
  public void feedForwardTest() {
    final INDArray[] activations = neuralNetwork.feedForward(input);
    assertTrue(Nd4jUtils.equals(activations[activations.length - 1], expectedOutput, tolerance));
    assertTrue(Nd4jUtils.equals(activations, expectedActivations, tolerance));
  }

  /**
   * Unit test for backprogation of neural network.
   */
  @Test
  public void backPropagateTest() {
    final INDArray[] activations = neuralNetwork.feedForward(input);
    assertTrue(Nd4jUtils.equals(activations, expectedActivations, tolerance));

    final INDArray[] errors = neuralNetwork.backPropagate(ArrayUtils.add(activations, 0, input), label);
    assertTrue(Nd4jUtils.equals(errors, expectedErrors, tolerance));
  }

  /**
   * Unit test for local neural network parameter provider.
   * @throws InjectionException
   */
  @Test
  public void localNeuralNetParameterProviderTest() throws InjectionException {
    final INDArray[] activations = ArrayUtils.add(expectedActivations, 0, input);
    final LocalNeuralNetParameterProvider localNeuralNetParameterProvider = Tang.Factory.getTang()
        .newInjector(neuralNetworkConfiguration).getInstance(LocalNeuralNetParameterProvider.class);

    localNeuralNetParameterProvider.push(neuralNetwork.generateParameterGradients(activations, expectedErrors));
    assertTrue(Nd4jUtils.equals(localNeuralNetParameterProvider.pull(), expectedParams, tolerance));
  }
}
