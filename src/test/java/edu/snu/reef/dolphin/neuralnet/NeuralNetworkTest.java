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
package edu.snu.reef.dolphin.neuralnet;

import edu.snu.reef.dolphin.neuralnet.conf.FullyConnectedLayerConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.neuralnet.layerparam.provider.LocalNeuralNetParameterProvider;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Unit tests for neural network.
 */
public class NeuralNetworkTest {

  private final double tolerance = 1e-9;

  private final INDArray input = Nd4j.create(new double[]{77, 57, 30, 26, 75, 74, 87, 75});
  private final INDArray expectedOutput =
      Nd4j.create(new double[]{5.96001904648e-01, 5.54388443935e-01, 4.88766051729e-01});
  private final INDArray label = Nd4j.create(new double[]{0, 1, 0});
  private final int numHiddenUnits = 5;

  private final List<INDArray> expectedActivations = Arrays.asList(input,
      Nd4j.create(
          new double[]{5.03440834992e-01, 5.05097234769e-01, 5.02210400594e-01, 5.07340068629e-01, 4.98231862363e-01}),
      expectedOutput);

  private final Configuration neuralNetworkConfiguration = NeuralNetworkConfigurationBuilder.newConfigurationBuilder()
      .setBatchSize(1)
      .setStepSize(1e-2)
      .setParameterProviderClass(LocalNeuralNetParameterProvider.class)
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumInput(input.length())
              .setNumOutput(numHiddenUnits)
              .setInitWeight(0.0001)
              .setInitBias(0.0002)
              .setRandomSeed(10)
              .setActivationFunction("sigmoid")
              .build())
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumInput(numHiddenUnits)
              .setNumOutput(expectedOutput.length())
              .setInitWeight(0.2)
              .setInitBias(0.3)
              .setRandomSeed(10)
              .setActivationFunction("sigmoid")
              .build())
      .build();

  private NeuralNetwork neuralNetwork;

  private final List<INDArray> expectedGradients = Arrays.asList(
      Nd4j.create(new double[]{
          -1.10814514935e-02, 4.75458113254e-02, 2.79511566851e-02, -3.76325218465e-02, -6.66430042946e-02}),
      Nd4j.create(new double[]{5.96001904648e-01, -4.45611556065e-01, 4.88766051729e-01}));

  private final LayerParameter[] expectedParams = new LayerParameter[]{
      LayerParameter.newBuilder()
          .setWeightParam(Nd4j.create(new double[]{
              8.43271085004e-03, 6.30096936863e-03, 3.39385184743e-03, 2.77477924060e-03, 8.53602946125e-03,
              8.44699779920e-03, 9.53429478040e-03, 8.27071991212e-03, -3.66582312238e-02, -2.70532006510e-02,
              -1.43677246544e-02, -1.24133925477e-02, -3.55010755578e-02, -3.51333945502e-02, -4.12985268766e-02,
              -3.56029583595e-02, -2.14895112802e-02, -1.59334007414e-02, -8.38015241706e-03, -7.31539494329e-03,
              -2.08851161497e-02, -2.06472822008e-02, -2.44371878001e-02, -2.08418701603e-02, 2.91009853375e-02,
              2.15053585334e-02, 1.12523984594e-02, 9.75559048423e-03, 2.83249945339e-02, 2.78617690825e-02,
              3.28448331050e-02, 2.82338715978e-02, 5.12505138331e-02, 3.79973748138e-02, 1.96798049550e-02,
              1.72828290869e-02, 5.00290409316e-02, 4.93010379396e-02, 5.80023241160e-02, 5.00251904698e-02})
              .reshape(input.length(), numHiddenUnits))
          .setBiasParam(Nd4j.create(new double[]{
              3.10814514935e-04, -2.75458113254e-04, -7.95115668513e-05, 5.76325218465e-04, 8.66430042946e-04})
              .reshape(1, numHiddenUnits))
          .build(),
      LayerParameter.newBuilder()
          .setWeightParam(Nd4j.create(new double[]{
              -2.03014116811e-01, 2.44876642191e-01, 9.28304253913e-02, 1.87009753641e-02, 7.41970556867e-03,
              -9.36696160145e-02, -1.26948175865e-01, -2.44954097078e-04, 1.41093564760e-01, -7.24960104289e-02,
              6.32980868345e-02, -3.33847090889e-02, 1.07187527879e-01, -2.10442219526e-01, -6.28627854949e-01})
              .reshape(numHiddenUnits, expectedOutput.length()))
          .setBiasParam(Nd4j.create(new double[]{2.94039980954e-01, 3.04456115561e-01, 2.95112339483e-01})
                  .reshape(1, expectedOutput.length()))
          .build()};

  @Before
  public void buildNeuralNetwork() throws InjectionException {
    final Injector injector = Tang.Factory.getTang().newInjector(neuralNetworkConfiguration);
    neuralNetwork = injector.getInstance(NeuralNetwork.class);
  }

  private boolean compare(final INDArray a, final INDArray b) {
    if (!Arrays.equals(a.shape(), b.shape())) {
      return false;
    }
    for (int i = 0; i < a.rows(); ++i) {
      for (int j = 0; j < a.columns(); ++j) {
        if (Math.abs(a.getDouble(i, j) - b.getDouble(i, j)) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  private boolean compare(final List<INDArray> a, final List<INDArray> b) {
    if (a.size() != b.size()) {
      return false;
    }
    final Iterator bIter = b.iterator();
    for (final INDArray m : a) {
      if (!compare(m, (INDArray) bIter.next())) {
        return false;
      }
    }
    return true;
  }

  private boolean compareParameters(final LayerParameter[] a, final LayerParameter[] b) {
    if (a.length != b.length) {
      return false;
    }
    for (int i = 0; i < a.length; ++i) {
      final LayerParameter param = a[i];
      final LayerParameter other = b[i];
      if (!compare(param.getBiasParam(), other.getBiasParam())
          || !compare(param.getWeightParam(), other.getWeightParam())) {
        return false;
      }
    }
    return true;
  }

  private void print(final INDArray matrix) {
    final int[] shape = matrix.shape();
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        System.out.print("\t" + matrix.getDouble(i, j));
      }
      System.out.println();
    }
  }

  /**
   * Unit test for feed forward of neural network.
   */
  @Test
  public void feedForwardTest() {
    final List<INDArray> activations = neuralNetwork.feedForward(input);
    assertTrue(compare(activations.get(activations.size() - 1), expectedOutput));
  }

  /**
   * Unit test for backprogation of neural network.
   */
  @Test
  public void backPropagateTest() {
    final Pair<List<INDArray>, List<INDArray>> actAndDeriv = neuralNetwork.activationAndDerivative(input);
    final List<INDArray> activations = actAndDeriv.first;
    final List<INDArray> derivatives = actAndDeriv.second;
    assertTrue(compare(activations, expectedActivations));

    final List<INDArray> gradients = neuralNetwork.backPropagate(activations, derivatives, label);
    assertTrue(compare(gradients, expectedGradients));
  }

  /**
   * Unit test for local neural network paramter provider.
   * @throws InjectionException
   */
  @Test
  public void localNeuralNetParameterProviderTest() throws InjectionException {
    final LocalNeuralNetParameterProvider localNeuralNetParameterProvider =
        Tang.Factory.getTang().newInjector(neuralNetworkConfiguration)
            .getInstance(LocalNeuralNetParameterProvider.class);

    localNeuralNetParameterProvider.push(expectedActivations, expectedGradients);
    assertTrue(compareParameters(localNeuralNetParameterProvider.pull(), expectedParams));
  }
}
