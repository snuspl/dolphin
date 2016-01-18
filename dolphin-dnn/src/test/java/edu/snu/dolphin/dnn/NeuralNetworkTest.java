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

import edu.snu.dolphin.dnn.blas.Matrix;
import edu.snu.dolphin.dnn.blas.MatrixFactory;
import edu.snu.dolphin.dnn.blas.jblas.MatrixJBLASFactory;
import edu.snu.dolphin.dnn.blas.MatrixUtils;
import edu.snu.dolphin.dnn.conf.ActivationLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.ActivationWithLossLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.FullyConnectedLayerConfigurationBuilder;
import edu.snu.dolphin.dnn.conf.NeuralNetworkConfigurationBuilder;
import edu.snu.dolphin.dnn.layers.LayerParameter;
import edu.snu.dolphin.dnn.layerparam.provider.LocalNeuralNetParameterProvider;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Injector;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.exceptions.InjectionException;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for neural network.
 */
public final class NeuralNetworkTest {

  private static MatrixFactory matrixFactory;
  private static LayerParameter emptyLayerParam;
  private static final float TOLERANCE = 1e-7f;

  static {
    final Configuration configuration = Tang.Factory.getTang().newConfigurationBuilder()
        .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
        .build();
    try {
      matrixFactory = Tang.Factory.getTang().newInjector(configuration).getInstance(MatrixFactory.class);
      emptyLayerParam = LayerParameter.newEmptyInstance(matrixFactory);
    } catch (final InjectionException e) {
      throw new RuntimeException("InjectionException while injecting a matrix factory: " + e);
    }
  }

  private final Matrix input = matrixFactory.create(new float[]{77, 57, 30, 26, 75, 74, 87, 75});
  private final Matrix expectedOutput =
      matrixFactory.create(new float[]{5.96001906923e-01f, 5.54388444438e-01f, 4.88766051614e-01f});
  private final Matrix label = matrixFactory.create(new float[]{0, 1, 0});
  private final int numHiddenUnits = 5;

  private final Matrix[] expectedActivations = new Matrix[] {
      matrixFactory.create(new float[]{
          1.37635586396e-02f, 2.03896453321e-02f, 8.84165966865e-03f, 2.93623840734e-02f, -7.07257980630e-03f}),
      matrixFactory.create(new float[]{
          5.03440835342e-01f, 5.05097234742e-01f, 5.02210400517e-01f, 5.07340068673e-01f, 4.98231862419e-01f}),
      matrixFactory.create(new float[]{
          3.88833708754e-01f, 2.18417983427e-01f, -4.49433571251e-02f}),
      expectedOutput};

  private final Configuration neuralNetworkConfiguration = NeuralNetworkConfigurationBuilder.newConfigurationBuilder()
      .setInputShape(input.getLength())
      .setStepsize(1e-2f)
      .setParameterProviderClass(LocalNeuralNetParameterProvider.class)
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumOutput(numHiddenUnits)
              .setInitWeight(0.0001f)
              .setInitBias(0.0002f)
              .setRandomSeed(10)
              .build())
      .addLayerConfiguration(
          ActivationLayerConfigurationBuilder.newConfigurationBuilder()
              .setActivationFunction("sigmoid")
              .build())
      .addLayerConfiguration(
          FullyConnectedLayerConfigurationBuilder.newConfigurationBuilder()
              .setNumOutput(expectedOutput.getLength())
              .setInitWeight(0.2f)
              .setInitBias(0.3f)
              .setRandomSeed(10)
              .build())
      .addLayerConfiguration(ActivationWithLossLayerConfigurationBuilder.newConfigurationBuilder()
          .setActivationFunction("sigmoid")
          .setLossFunction("crossentropy")
          .build())
      .build();

  private final Configuration blasConfiguration = Tang.Factory.getTang().newConfigurationBuilder()
      .bindImplementation(MatrixFactory.class, MatrixJBLASFactory.class)
      .build();

  private NeuralNetwork neuralNetwork;

  private final Matrix[] expectedErrors = new Matrix[] {
      matrixFactory.create(new float[]{
          -1.10814502938e-02f, 4.75458121280e-02f, 2.79511566348e-02f, -3.76325213356e-02f, -6.66430044053e-02f}),
      matrixFactory.create(new float[]{
          -4.43279004290e-02f, 1.90203015780e-01f, 1.11806811634e-01f, -1.50562532537e-01f, -2.66575351211e-01f}),
      matrixFactory.create(new float[]{5.96001906923e-01f, -4.45611555562e-01f, 4.88766051614e-01f})};

  private final LayerParameter[] expectedParams = new LayerParameter[]{
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[]{
              8.43270993132e-03f, -3.66582318410e-02f, -2.14895112413e-02f, 2.91009849480e-02f, 5.12505139198e-02f,
              6.30096868417e-03f, -2.70532011080e-02f, -1.59334007127e-02f, 2.15053582413e-02f, 3.79973748771e-02f,
              3.39385148596e-03f, -1.43677248949e-02f, -8.38015240206e-03f, 1.12523983074e-02f, 1.96798049890e-02f,
              2.77477892309e-03f, -1.24133927578e-02f, -7.31539492911e-03f, 9.75559034996e-03f, 1.72828291146e-02f,
              8.53602856872e-03f, -3.55010761603e-02f, -2.08851161142e-02f, 2.83249941530e-02f, 5.00290410160e-02f,
              8.44699691700e-03f, -3.51333951458e-02f, -2.06472821629e-02f, 2.78617687040e-02f, 4.93010380208e-02f,
              9.53429374101e-03f, -4.12985275728e-02f, -2.44371877617e-02f, 3.28448326583e-02f, 5.80023242130e-02f,
              8.27071901140e-03f, -3.56029589638e-02f, -2.08418701192e-02f, 2.82338712143e-02f, 5.00251905529e-02f},
              numHiddenUnits, input.getLength()))
          .setBiasParam(matrixFactory.create(new float[]{
              3.10814502938e-04f, -2.75458121280e-04f, -7.95115663479e-05f, 5.76325213356e-04f, 8.66430044053e-04f}))
          .build(),
      emptyLayerParam, // sigmoid activation layer
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[]{
              -2.03014106838e-01f, -9.36696143375e-02f, 6.32980870484e-02f, 2.44876650034e-01f, -1.26948172924e-01f,
              -3.33847104410e-02f, 9.28304262651e-02f, -2.44954075032e-04f, 1.07187525993e-01f, 1.87009757363e-02f,
              1.41093561592e-01f, -2.10442218992e-01f, 7.41970535523e-03f, -7.24960077710e-02f, -6.28627853302e-01f},
              expectedOutput.getLength(), numHiddenUnits))
          .setBiasParam(matrixFactory.create(new float[]{2.94039980931e-01f, 3.04456115556e-01f, 2.95112339484e-01f}))
          .build(),
      emptyLayerParam}; // sigmoid activation layer

  @Before
  public void buildNeuralNetwork() throws InjectionException {
    final Injector injector = Tang.Factory.getTang().newInjector(blasConfiguration, neuralNetworkConfiguration);
    neuralNetwork = injector.getInstance(NeuralNetwork.class);
  }

  /**
   * Unit test for feedforward of neural network.
   */
  @Test
  public void feedForwardTest() {
    final Matrix[] activations = neuralNetwork.feedForward(input);
    assertTrue(expectedOutput.compare(activations[activations.length - 1], TOLERANCE));
    assertTrue(MatrixUtils.compare(activations, expectedActivations, TOLERANCE));
  }

  /**
   * Unit test for backprogation of neural network.
   */
  @Test
  public void backPropagateTest() {
    final Matrix[] activations = neuralNetwork.feedForward(input);
    assertTrue(MatrixUtils.compare(activations, expectedActivations, TOLERANCE));

    final Matrix[] gradients = neuralNetwork.backPropagate(ArrayUtils.add(activations, 0, input), label);
    assertTrue(MatrixUtils.compare(expectedErrors, gradients, TOLERANCE));
  }

  /**
   * Returns true if each element of weight and bias of a layer parameter is equal to another within tolerance.
   *
   * @param a one layer parameter array to be tested for equality.
   * @param b another layer parameter array to be tested for equality.
   * @param tolerance the maximum difference for which both numbers are still considered equal.
   * @return true if two layer parameter arrays are equal.
   */
  private static boolean compare(final LayerParameter[] a, final LayerParameter[] b, final float tolerance) {
    if (a.length != b.length) {
      return false;
    }
    for (int i = 0; i < a.length; ++i) {
      final LayerParameter param = a[i];
      final LayerParameter other = b[i];
      if (!param.getBiasParam().compare(other.getBiasParam(), tolerance)
          || !param.getWeightParam().compare(other.getWeightParam(), tolerance)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Unit test for local neural network parameter provider.
   * @throws InjectionException
   */
  @Test
  public void localNeuralNetParameterProviderTest() throws InjectionException {
    final Matrix[] activations = ArrayUtils.add(expectedActivations, 0, input);
    final LocalNeuralNetParameterProvider localNeuralNetParameterProvider = Tang.Factory.getTang()
        .newInjector(blasConfiguration, neuralNetworkConfiguration).getInstance(LocalNeuralNetParameterProvider.class);

    localNeuralNetParameterProvider.push(input.getColumns(),
        neuralNetwork.generateParameterGradients(activations, expectedErrors));
    assertTrue(compare(expectedParams, localNeuralNetParameterProvider.pull(), TOLERANCE));
  }

  private static final int numBatch = 3;

  private final Matrix batchInput = matrixFactory.create(new float[]{
      77, 57, 30, 26, 75, 74, 87, 75,
      61, 5, 18, 18, 16, 4, 67, 29,
      68, 85, 4, 50, 19, 3, 5, 18}, input.getLength(), numBatch);

  private final Matrix expectedBatchOutput = matrixFactory.create(new float[]{
      5.96001906923e-01f, 5.54388444438e-01f, 4.88766051614e-01f,
      5.95962765220e-01f, 5.54549180394e-01f, 4.88787332887e-01f,
      5.95960592285e-01f, 5.54529568429e-01f, 4.88781434177e-01f}, expectedOutput.getLength(), numBatch);

  private final Matrix labels = matrixFactory.create(new float[]{
      0, 1, 0,
      0, 0, 1,
      1, 0, 0}, expectedOutput.getLength(), numBatch);

  private final Matrix[] expectedBatchActivations = new Matrix[] {
      matrixFactory.create(new float[]{
          1.37635586396e-02f, 2.03896453321e-02f, 8.84165966865e-03f, 2.93623840734e-02f, -7.07257980630e-03f,
          -1.03681771740e-02f, 3.53007655740e-03f, -1.66967848856e-03f, 1.57861485705e-02f, -6.65068837926e-03f,
          -9.20206232816e-03f, 2.52719653249e-03f, 3.13138561007e-03f, 1.43411668226e-02f, -5.00741667077e-03f},
          numHiddenUnits, numBatch),
      matrixFactory.create(new float[]{
          5.03440835342e-01f, 5.05097234742e-01f, 5.02210400517e-01f, 5.07340068673e-01f, 4.98231862419e-01f,
          4.97407978926e-01f, 5.00882518223e-01f, 4.99582580475e-01f, 5.03946455187e-01f, 4.98337334034e-01f,
          4.97699500651e-01f, 5.00631798797e-01f, 5.00782845763e-01f, 5.03585230258e-01f, 4.98748148448e-01f},
          numHiddenUnits, numBatch),
      matrixFactory.create(new float[]{
          3.88833708754e-01f, 2.18417983427e-01f, -4.49433571251e-02f,
          3.88671151640e-01f, 2.19068648969e-01f, -4.48581891244e-02f,
          3.88662127499e-01f, 2.18989256483e-01f, -4.48817958414e-02f},
          expectedOutput.getLength(), numBatch),
      expectedBatchOutput};

  private final Matrix[] expectedBatchErrors = new Matrix[] {
      matrixFactory.create(new float[]{
          -1.10814502938e-02f, 4.75458121280e-02f, 2.79511566348e-02f, -3.76325213356e-02f, -6.66430044053e-02f,
          -5.15000730878e-02f, 2.29721560018e-02f, -8.00065487467e-05f, 4.90593973619e-02f, 7.12180587144e-02f,
          1.49417896767e-02f, -4.67279048839e-02f, 3.37442108292e-03f, -8.35931344072e-03f, -8.79247789398e-02f},
          numHiddenUnits, numBatch),
      matrixFactory.create(new float[]{
          -4.43279004290e-02f, 1.90203015780e-01f, 1.11806811634e-01f, -1.50562532537e-01f, -2.66575351211e-01f,
          -2.06005828612e-01f, 9.18889102737e-02f, -3.20026418031e-04f, 1.96249815425e-01f, 2.84875384962e-01f,
          5.97684239558e-02f, -1.86911917974e-01f, 1.34977174198e-02f, -3.34389730445e-02f, -3.51701320409e-01f},
          numHiddenUnits, numBatch),
      matrixFactory.create(new float[]{
          5.96001906923e-01f, -4.45611555562e-01f, 4.88766051614e-01f,
          5.95962765220e-01f, 5.54549180394e-01f, -5.11212667113e-01f,
          -4.04039407715e-01f, 5.54529568429e-01f, 4.88781434177e-01f},
          expectedOutput.getLength(), numBatch)};

  /**
   * Unit test for feedforward of neural network for a batch input.
   */
  @Test
  public void feedForwardTestForBatch() {
    final Matrix[] batchActivations = neuralNetwork.feedForward(batchInput);
    assertTrue(expectedBatchOutput.compare(batchActivations[batchActivations.length - 1], TOLERANCE));
    assertTrue(MatrixUtils.compare(batchActivations, expectedBatchActivations, TOLERANCE));
  }

  /**
   * Unit test for backpropagate of neural network for a batch input.
   */
  @Test
  public void backPropagateTestForBatch() {
    final Matrix[] batchActivations = ArrayUtils.add(expectedBatchActivations, 0, batchInput);
    final Matrix[] gradients = neuralNetwork.backPropagate(batchActivations, labels);
    assertTrue(MatrixUtils.compare(expectedBatchErrors, gradients, TOLERANCE));
  }

  private final LayerParameter[] expectedBatchParams = new LayerParameter[]{
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[]{
              9.82910798162e-03f, -6.33072822863e-03f, -7.88985161601e-03f, 1.70232424537e-03f, 2.24890496128e-02f,
              -1.28515495108e-03f, 3.87091128434e-03f, -6.26671372245e-03f, 8.75581565258e-03f, 3.63980862575e-02f,
              4.06834195010e-03f, -5.61385309760e-03f, -2.83011296328e-03f, 8.93787711127e-04f, 3.25045097098e-03f,
              1.45370031131e-03f, 2.23753613154e-03f, -3.02813089998e-03f, 1.68227505019e-03f, 1.61124213181e-02f,
              4.79566064030e-03f, -9.99391777382e-03f, -7.11898411617e-03f, 7.42165547717e-03f, 1.84778116814e-02f,
              3.51739784978e-03f, -1.15164775272e-02f, -6.89072234663e-03f, 8.72585934798e-03f, 1.63534961874e-02f,
              1.43597057322e-02f, -1.80736062976e-02f, -8.26388913572e-03f, 2.00693430198e-04f, 4.90942819402e-03f,
              6.81182688237e-03f, -1.12470203536e-02f, -7.06102310040e-03f, 5.17676094132e-03f, 1.50947627443e-02f},
              numHiddenUnits, input.getLength()))
          .setBiasParam(matrixFactory.create(new float[]{
              3.58799112350e-04f, 1.20699789180e-04f, 9.58480961035e-05f, 1.89774791381e-04f, 4.77832415436e-04f}))
          .build(),
      emptyLayerParam, // sigmoid activation layer
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[]{
              -2.01331583596e-01f, -9.70046289504e-02f, 6.49752355899e-02f, 2.46562801617e-01f, -1.30299951105e-01f,
              -3.17010213689e-02f, 9.45078932686e-02f, -3.58603247992e-03f, 1.08859347243e-01f, 2.03939299096e-02f,
              1.37723997630e-01f, -2.08750812643e-01f, 9.08109765882e-03f, -7.58192041248e-02f, -6.26967802760e-01f},
              expectedOutput.getLength(), numHiddenUnits))
          .setBiasParam(matrixFactory.create(new float[]{2.97373582452e-01f, 2.97788442689e-01f, 2.98445550604e-01f}))
          .build(),
      emptyLayerParam}; // sigmoid activation layer

  /**
   * Unit test for local neural network parameter provider for a batch input.
   * @throws InjectionException
   */
  @Test
  public void localNeuralNetParameterProviderBatchTest() throws InjectionException {
    final Matrix[] batchActivations = ArrayUtils.add(expectedBatchActivations, 0, batchInput);
    final LocalNeuralNetParameterProvider localNeuralNetParameterProvider = Tang.Factory.getTang()
        .newInjector(blasConfiguration, neuralNetworkConfiguration).getInstance(LocalNeuralNetParameterProvider.class);

    localNeuralNetParameterProvider.push(batchInput.getColumns(),
        neuralNetwork.generateParameterGradients(batchActivations, expectedBatchErrors));
    assertTrue(compare(expectedBatchParams, localNeuralNetParameterProvider.pull(), TOLERANCE));
  }
}
