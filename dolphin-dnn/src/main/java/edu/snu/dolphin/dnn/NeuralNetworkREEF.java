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

import edu.snu.dolphin.bsp.parameters.*;
import edu.snu.dolphin.dnn.data.NeuralNetParamServerDataCodec;
import edu.snu.dolphin.dnn.data.NeuralNetParamWorkerDataCodec;
import edu.snu.dolphin.ps.ParameterServerConfigurationBuilder;
import edu.snu.dolphin.ps.driver.SingleNodeParameterServerManager;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.reef.client.DriverConfiguration;
import org.apache.reef.client.DriverLauncher;
import org.apache.reef.client.LauncherStatus;
import org.apache.reef.driver.evaluator.EvaluatorRequest;
import org.apache.reef.io.data.loading.api.DataLoadingRequestBuilder;
import org.apache.reef.io.network.group.impl.driver.GroupCommService;
import org.apache.reef.runtime.local.client.LocalRuntimeConfiguration;
import org.apache.reef.runtime.yarn.client.YarnClientConfiguration;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.JavaConfigurationBuilder;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;
import org.apache.reef.tang.exceptions.InjectionException;
import org.apache.reef.tang.formats.CommandLine;
import org.apache.reef.tang.formats.ConfigurationModule;
import org.apache.reef.util.EnvironmentUtils;

import javax.inject.Inject;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Job launch code for the neural network REEF job.
 */
public final class NeuralNetworkREEF {

  private static final Logger LOG = Logger.getLogger(NeuralNetworkREEF.class.getName());

  private final int evalSize;
  private final boolean onLocal;
  private final int timeout;
  private final String inputDir;
  private final int desiredSplits;
  private final NeuralNetworkDriverParameters neuralNetworkDriverParameters;

  @Inject
  private NeuralNetworkREEF(@Parameter(EvaluatorSize.class) final int evalSize,
                            @Parameter(OnLocal.class) final boolean onLocal,
                            @Parameter(Timeout.class) final int timeout,
                            @Parameter(InputDir.class) final String inputDir,
                            @Parameter(DesiredSplits.class) final int desiredSplits,
                            final NeuralNetworkDriverParameters neuralNetworkDriverParameters) {
    this.evalSize = evalSize;
    this.onLocal = onLocal;
    this.timeout = timeout;
    this.inputDir = inputDir;
    this.desiredSplits = desiredSplits;
    this.neuralNetworkDriverParameters = neuralNetworkDriverParameters;
  }

  /**
   * Parses command line parameters and instantiate the neural network REEF job.
   * @param args command line arguments.
   * @return the neural network REEF job instance.
   * @throws IOException
   * @throws InjectionException
   */
  private static NeuralNetworkREEF parseCommandLine(final String[] args) throws IOException, InjectionException {
    final JavaConfigurationBuilder cb = Tang.Factory.getTang().newConfigurationBuilder();
    final CommandLine cl = new CommandLine(cb);

    cl.registerShortNameOfClass(EvaluatorSize.class);
    cl.registerShortNameOfClass(OnLocal.class);
    cl.registerShortNameOfClass(Timeout.class);
    cl.registerShortNameOfClass(InputDir.class);
    cl.registerShortNameOfClass(DesiredSplits.class);
    NeuralNetworkDriverParameters.registerShortNameOfClass(cl);

    cl.processCommandLine(args);

    return Tang.Factory.getTang().newInjector(cb.build()).getInstance(NeuralNetworkREEF.class);
  }

  /**
   * @return the configuration for running a neural network job.
   */
  private Configuration getRuntimeConfiguration() {
    return onLocal ? getLocalRuntimeConfiguration() : getYarnRuntimeConfiguration();
  }

  /**
   * Builds and returns the configuration for the Driver.
   * The configuration changes depending on whether we use REEF Group Communication or not.
   * TODO #68: This code may change when asynchronous Parameter Server is introduced.
   *
   * @return the configuration for driver with data loading.
   */
  private Configuration getDriverConfWithDataLoad() {
    switch (neuralNetworkDriverParameters.getProviderType()) {
    case LOCAL:
      return getDriverConfLocalProvider();
    case GROUP_COMM:
      return getDriverConfGroupCommProvider();
    case PARAMETER_SERVER:
      return getDriverConfParamServerProvider();
    default:
      throw new RuntimeException("No driver configuration for provider type " +
          neuralNetworkDriverParameters.getProviderType());
    }
  }

  private Configuration getDriverConfLocalProvider() {
    final ConfigurationModule neuralNetworkDriverConf = DriverConfiguration.CONF
        .set(DriverConfiguration.GLOBAL_LIBRARIES, EnvironmentUtils.getClassLocation(NeuralNetworkDriver.class))
        .set(DriverConfiguration.GLOBAL_LIBRARIES, EnvironmentUtils.getClassLocation(TextInputFormat.class))
        .set(DriverConfiguration.DRIVER_IDENTIFIER, "NeuralNetworkLocal")
        .set(DriverConfiguration.ON_CONTEXT_ACTIVE, NeuralNetworkDriver.ActiveContextHandler.class);

    final EvaluatorRequest dataRequest = EvaluatorRequest.newBuilder()
        .setNumberOfCores(1)
        .setMemory(evalSize)
        .build();

    final Configuration dataLoadingConfiguration = new DataLoadingRequestBuilder()
        .setInputFormatClass(TextInputFormat.class)
        .setInputPath(processInputDir(inputDir))
        .setNumberOfDesiredSplits(desiredSplits)
        .addDataRequest(dataRequest)
        .setDriverConfigurationModule(neuralNetworkDriverConf)
        .build();

    return Configurations.merge(
        dataLoadingConfiguration,
        neuralNetworkDriverParameters.getDriverConfiguration());
  }

  private Configuration getDriverConfGroupCommProvider() {
    final ConfigurationModule neuralNetworkDriverConf = DriverConfiguration.CONF
        .set(DriverConfiguration.GLOBAL_LIBRARIES,
            EnvironmentUtils.getClassLocation(NeuralNetworkGroupCommDriver.class))
        .set(DriverConfiguration.GLOBAL_LIBRARIES, EnvironmentUtils.getClassLocation(TextInputFormat.class))
        .set(DriverConfiguration.DRIVER_IDENTIFIER, "NeuralNetworkGroupComm")
        .set(DriverConfiguration.ON_CONTEXT_ACTIVE, NeuralNetworkGroupCommDriver.ActiveContextHandler.class);

    final Configuration dataLoadingConfiguration = new DataLoadingRequestBuilder()
        .setInputFormatClass(TextInputFormat.class)
        .setInputPath(processInputDir(inputDir))
        .setNumberOfDesiredSplits(desiredSplits)
        .addDataRequest(EvaluatorRequest.newBuilder()
            .setNumberOfCores(1)
            .setMemory(evalSize)
            .build())
        .addComputeRequest(EvaluatorRequest.newBuilder()
            .setNumber(1)
            .setNumberOfCores(1)
            .setMemory(evalSize)
            .build())
        .setDriverConfigurationModule(neuralNetworkDriverConf)
        .build();

    return Configurations.merge(
        dataLoadingConfiguration,
        neuralNetworkDriverParameters.getDriverConfiguration(),
        GroupCommService.getConfiguration());
  }

  private Configuration getDriverConfParamServerProvider() {
    final ConfigurationModule neuralNetworkDriverConf = DriverConfiguration.CONF
        .set(DriverConfiguration.GLOBAL_LIBRARIES,
            EnvironmentUtils.getClassLocation(NeuralNetworkSingleNodeParameterServerDriver.class))
        .set(DriverConfiguration.GLOBAL_LIBRARIES,
            EnvironmentUtils.getClassLocation(TextInputFormat.class))
        .set(DriverConfiguration.DRIVER_IDENTIFIER, "NeuralNetworkSingleNodeParamServer")
        .set(DriverConfiguration.ON_CONTEXT_ACTIVE,
            NeuralNetworkSingleNodeParameterServerDriver.ActiveContextHandler.class)
        .set(DriverConfiguration.ON_TASK_COMPLETED,
            NeuralNetworkSingleNodeParameterServerDriver.CompletedTaskHandler.class);

    final Configuration dataLoadingConfiguration = new DataLoadingRequestBuilder()
        .setInputFormatClass(TextInputFormat.class)
        .setInputPath(processInputDir(inputDir))
        .setNumberOfDesiredSplits(desiredSplits)
        .addDataRequest(EvaluatorRequest.newBuilder()
            .setNumberOfCores(1)
            .setMemory(evalSize)
            .build())
        .addComputeRequest(EvaluatorRequest.newBuilder()
            .setNumber(1)
            .setNumberOfCores(1)
            .setMemory(evalSize)
            .build())
        .setDriverConfigurationModule(neuralNetworkDriverConf)
        .build();

    final Configuration parameterServerConfiguration = new ParameterServerConfigurationBuilder()
        .setManagerClass(SingleNodeParameterServerManager.class)
        .setUpdaterClass(NeuralNetworkParameterUpdater.class)
        .setPreValueCodecClass(NeuralNetParamWorkerDataCodec.class)
        .setValueCodecClass(NeuralNetParamServerDataCodec.class)
        .build();

    return Configurations.merge(
        dataLoadingConfiguration,
        neuralNetworkDriverParameters.getDriverConfiguration(),
        parameterServerConfiguration);
  }

  /**
   * @return the configuration for running on local environment.
   */
  private Configuration getLocalRuntimeConfiguration() {
    return LocalRuntimeConfiguration.CONF
        .set(LocalRuntimeConfiguration.MAX_NUMBER_OF_EVALUATORS, desiredSplits + 1)
        .build();
  }

  /**
   * @return the configuration for running on YARN.
   */
  private Configuration getYarnRuntimeConfiguration() {
    return YarnClientConfiguration.CONF.build();
  }

  /**
   * Runs neural network REEF job.
   * @return the status of the launcher.
   * @throws InjectionException
   */
  private LauncherStatus run() throws InjectionException {
    return DriverLauncher.getLauncher(getRuntimeConfiguration()).run(getDriverConfWithDataLoad(), timeout);
  }

  /**
   * Changes the given path string to the string supported by REEF Data Loading service.
   * @param inputPath the input path string given by user.
   * @return the path string supported by REEF Data Loading service.
   */
  private String processInputDir(final String inputPath) {
    if (!onLocal) {
      return inputPath;
    }
    final File inputFile = new File(inputPath);
    return "file:///" + (inputFile.getAbsolutePath());
  }

  public static void main(final String[] args) {
    LauncherStatus status;
    try {
      final NeuralNetworkREEF neuralNetworkREEF = parseCommandLine(args);
      status = neuralNetworkREEF.run();
    } catch (final Exception e) {
      LOG.log(Level.SEVERE, "Fatal exception occurred: {0}", e);
      status = LauncherStatus.failed(e);
    }
    LOG.log(Level.INFO, "REEF job completed: {0}", status);
  }
}
