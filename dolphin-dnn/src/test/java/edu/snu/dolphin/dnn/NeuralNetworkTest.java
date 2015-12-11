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
      matrixFactory.create(new float[]{5.61329743164e-01f, 5.97883503916e-01f, 4.79822915984e-01f});
  private final Matrix label = matrixFactory.create(new float[]{0, 1, 0});
  private final int numHiddenUnits = 5;

  private final Matrix[] expectedActivations = new Matrix[] {
      matrixFactory.create(new float[]{
          -7.93876278605e-03f,  -3.31747899111e-02f, 3.44120437890e-02f, 1.59688640758e-02f, 1.38468652740e-02f}),
      matrixFactory.create(new float[]{
          4.98015319727e-01f, 4.91707063086e-01f, 5.08602162082e-01f, 5.03992131185e-01f, 5.03461661008e-01f}),
      matrixFactory.create(new float[]{
          2.46560502863e-01f, 3.96654087576e-01f, -8.07521889872e-02f}),
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
          -8.30651912736e-03f, -4.80258488777e-02f, 1.07207504661e-02f, 1.48007835060e-02f, -9.54255943309e-02f}),
      matrixFactory.create(new float[]{
          -3.32266000219e-02f, -1.92156256008e-01f, 4.28956985094e-02f, 5.92069083725e-02f, -3.81720674107e-01f}),
      matrixFactory.create(new float[]{5.61329743164e-01f, -4.02116496084e-01f, 4.79822915984e-01f})};

  private final LayerParameter[] expectedParams = new LayerParameter[]{
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[][]{
              {6.29601293314e-03f, 3.70347247158e-02f, -8.30645946347e-03f, -1.13498155877e-02f, 7.33580261453e-02f},
              {4.68675940016e-03f, 2.73855962264e-02f, -6.15892196974e-03f, -8.18972289887e-03f, 5.44971278649e-02f},
              {2.52483510568e-03f, 1.44771710611e-02f, -3.24509033713e-03f, -4.38972922295e-03f, 2.86505886796e-02f},
              {2.28363849271e-03f, 1.23827394517e-02f, -2.83174715194e-03f, -3.81162996477e-03f, 2.47702858171e-02f},
              {6.16528987324e-03f, 3.60245812466e-02f, -7.81562200122e-03f, -1.10868847139e-02f, 7.16255958803e-02f},
              {6.13136617093e-03f, 3.55017700762e-02f, -7.77507240929e-03f, -1.09673650336e-02f, 7.07364371617e-02f},
              {7.27458344572e-03f, 4.14693921910e-02f, -9.24880154366e-03f, -1.29832496649e-02f, 8.30297472805e-02f},
              {6.22864791469e-03f, 3.59129885050e-02f, -7.93995969825e-03f, -1.10342586510e-02f, 7.16121329971e-02f}}))
          .setBiasParam(matrixFactory.create(new float[]{
              2.83065191274e-04f, 6.80258488777e-04f, 9.27924953390e-05f, 5.19921649397e-05f, 1.15425594331e-03f}))
          .build(),
      emptyLayerParam, // sigmoid activation layer
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[][]{
              {-2.02809097974e-01f, -2.89133648763e-02f, 1.36443203991e-01f},
              {-9.86731028695e-02f, 9.78008450420e-02f, -2.10321836138e-01f},
              {6.29037997313e-02f, -4.37688459799e-04f, 7.94878703128e-03f},
              {2.45057981449e-01f, 1.11668795437e-01f, -7.71344562636e-02f},
              {-1.32025024617e-01f, 2.37492346105e-02f, -6.28608389523e-01f}}))
          .setBiasParam(matrixFactory.create(new float[]{2.94386702568e-01f, 3.04021164961e-01f, 2.95201770840e-01f}))
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

    localNeuralNetParameterProvider.push(input.getRows(),
        neuralNetwork.generateParameterGradients(activations, expectedErrors));
    assertTrue(compare(expectedParams, localNeuralNetParameterProvider.pull(), TOLERANCE));
  }

  private final Matrix batchInput = matrixFactory.create(new float[][]{
      {77, 57, 30, 26, 75, 74, 87, 75},
      {61, 5, 18, 18, 16, 4, 67, 29},
      {68, 85, 4, 50, 19, 3, 5, 18}});

  private final Matrix expectedBatchOutput = matrixFactory.create(new float[][]{
      {5.61329743164e-01f, 5.97883503916e-01f, 4.79822915984e-01f},
      {5.60976372061e-01f, 5.97821453723e-01f, 4.80461166777e-01f},
      {5.61245484844e-01f, 5.98112837378e-01f, 4.79875062394e-01f}});

  private final Matrix labels = matrixFactory.create(new float[][]{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}});

  private final Matrix[] expectedBatchActivations = new Matrix[] {
      matrixFactory.create(new float[][]{
          {-7.93876278605e-03f, -3.31747899111e-02f, 3.44120437890e-02f, 1.59688640758e-02f, 1.38468652740e-02f},
          {-1.23871909026e-03f, -2.11530894332e-02f, 7.89375894368e-03f, 7.98690381646e-04f, -3.62337928265e-03f},
          {-5.40462196656e-03f, -3.56428819858e-03f, -2.77098032534e-03f, 2.72608707532e-02f, 1.27705410570e-03f}}),
      matrixFactory.create(new float[][]{
          {4.98015319727e-01f, 4.91707063086e-01f, 5.08602162082e-01f, 5.03992131185e-01f, 5.03461661008e-01f},
          {4.99690320267e-01f, 4.94711924821e-01f, 5.01973429489e-01f, 5.00199672585e-01f, 4.99094156170e-01f},
          {4.98648847797e-01f, 4.99108928894e-01f, 4.99307255362e-01f, 5.06814795656e-01f, 5.00319263483e-01f}}),
      matrixFactory.create(new float[][]{
          {2.46560502863e-01f, 3.96654087576e-01f, -8.07521889872e-02f},
          {2.45125553286e-01f, 3.96396002015e-01f, -7.81951521132e-02f},
          {2.46218328523e-01f, 3.97608068197e-01f, -8.05432640010e-02f}}),
      expectedBatchOutput};

  private final Matrix[] expectedBatchErrors = new Matrix[] {
      matrixFactory.create(new float[][]{
          {-8.30651912736e-03f, -4.80258488777e-02f, 1.07207504661e-02f, 1.48007835060e-02f, -9.54255943309e-02f},
          {-5.07035192412e-02f, 2.78781517325e-02f, 7.50168509571e-03f, 6.08557822289e-02f, 6.64601224626e-02f},
          {3.39717583167e-02f, -1.00106875386e-04f, -6.33785444658e-03f, -1.97557316255e-02f, -5.77034221542e-02f}}),
      matrixFactory.create(new float[][]{
          {-3.32266000219e-02f, -1.92156256008e-01f, 4.28956985094e-02f, 5.92069083725e-02f, -3.81720674107e-01f},
          {-2.02814154765e-01f, 1.11525081563e-01f, 3.00072078260e-02f, 2.43423167736e-01f, 2.65841362398e-01f},
          {1.35888025582e-01f, -4.00428773318e-04f, -2.53514664505e-02f, -7.90376089833e-02f, -2.30813782723e-01f}}),
      matrixFactory.create(new float[][]{
          {5.61329743164e-01f, -4.02116496084e-01f, 4.79822915984e-01f},
          {5.60976372061e-01f, 5.97821453723e-01f, -5.19538833223e-01f},
          {-4.38754515156e-01f, 5.98112837378e-01f, 4.79875062394e-01f}})};

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
          .setWeightParam(matrixFactory.create(new float[][]{
              {4.64145014168e-03f, 6.73558899806e-03f, -2.89190318578e-03f, -1.16481232727e-02f, 2.39387718430e-02f},
              {-7.24999073729e-03f, 8.69950140536e-03f, -4.14339451025e-04f, 2.01776909015e-03f, 3.34770362552e-02f},
              {3.45278565712e-03f, 3.20064693997e-03f, -1.46653662370e-03f, -4.81750970047e-03f, 6.34724142775e-03f},
              {-1.77590672102e-03f, 2.40225435486e-03f, -3.67275769129e-04f, -1.60488581988e-03f, 1.38594791443e-02f},
              {2.56467330903e-03f, 1.05311621508e-02f, -2.45393919166e-03f, -5.68093834349e-03f, 2.40228089200e-02f},
              {2.36981274148e-03f, 1.14383103422e-02f, -2.52281276950e-03f, -4.27949828405e-03f, 2.33507098805e-02f},
              {1.32143923438e-02f, 7.38994773632e-03f, -4.60051170392e-03f, -1.76606577354e-02f, 1.38018655878e-02f},
              {4.93842304532e-03f, 9.21118247783e-03f, -2.92447609099e-03f, -8.33124861590e-03f, 2.09370626562e-02f}}))
          .setBiasParam(matrixFactory.create(new float[]{
              2.83460933506e-04f, 2.67492680069e-04f, 1.60384729616e-04f, 1.36638863018e-05f, 4.88896313408e-04f}))
          .build(),
      emptyLayerParam, // sigmoid activation layer
      LayerParameter.newBuilder()
          .setWeightParam(matrixFactory.create(new float[][]{
              {-2.01150525996e-01f, -3.22383456150e-02f, 1.38103996340e-01f},
              {-9.70281555556e-02f, 9.45017787010e-02f, -2.08690580267e-01f},
              {6.45986834694e-02f, -3.79691247050e-03f, 9.64634547967e-03f},
              {2.46749910214e-01f, 1.08310496669e-01f, -7.54667251949e-02f},
              {-1.30342513562e-01f, 2.04075111228e-02f, -6.26933879715e-01f}}))
          .setBiasParam(matrixFactory.create(new float[]{2.97721494666e-01f, 2.97353940683e-01f, 2.98532802849e-01f}))
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

    localNeuralNetParameterProvider.push(batchInput.getRows(),
        neuralNetwork.generateParameterGradients(batchActivations, expectedBatchErrors));
    assertTrue(compare(expectedBatchParams, localNeuralNetParameterProvider.pull(), TOLERANCE));
  }
}
