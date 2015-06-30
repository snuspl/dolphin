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
package edu.snu.reef.dolphin.core;

import edu.snu.reef.dolphin.parameters.*;
import org.apache.reef.tang.Configuration;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.io.File;

/**
 * Parameters required by Dolphin Launcher.
 */
public final class DolphinParameters {
  private final String identifier;
  private final UserJobInfo userJobInfo;
  private final UserParameters userParameters;
  private final int evalNum;
  private final int evalSize;
  private final String inputDir;
  private final String outputDir;
  private final boolean onLocal;
  private final int timeout;

  /**
   * Dolphin Parameters constructor - instantiated by Tang.
   *
   * @param identifier
   * @param userJobInfo
   * @param userParameters
   * @param evalNum
   * @param evalSize
   * @param inputDir
   * @param outputDir
   * @param onLocal
   * @param timeout
   */
  @Inject
  private DolphinParameters(@Parameter(JobIdentifier.class) final String identifier,
                            final UserJobInfo userJobInfo,
                            final UserParameters userParameters,
                            @Parameter(EvaluatorNum.class) final int evalNum,
                            @Parameter(EvaluatorSize.class) final int evalSize,
                            @Parameter(InputDir.class) final String inputDir,
                            @Parameter(OutputDir.class) final String outputDir,
                            @Parameter(OnLocal.class) final boolean onLocal,
                            @Parameter(Timeout.class) final int timeout) {
    this.identifier = identifier;
    this.userJobInfo = userJobInfo;
    this.userParameters = userParameters;
    this.evalNum = evalNum;
    this.evalSize = evalSize;
    this.inputDir = inputDir;
    this.outputDir = outputDir;
    this.onLocal = onLocal;
    this.timeout = timeout;
  }

  /**
   * Return a configuration for the driver.
   * @return
   */
  public Configuration getDriverConf() {
    Configuration driverConf = Tang.Factory.getTang().newConfigurationBuilder()
        .bindNamedParameter(EvaluatorNum.class, String.valueOf(evalNum))
        .bindNamedParameter(OutputDir.class, processOutputDir(outputDir, onLocal))
        .bindNamedParameter(OnLocal.class, String.valueOf(onLocal))
        .bindImplementation(UserJobInfo.class, userJobInfo.getClass())
        .bindImplementation(UserParameters.class, userParameters.getClass())
        .build();
    return Configurations.merge(userParameters.getDriverConf(), driverConf);
  }

  /**
   * Process the path of the output directory.
   * If a relative local file path is given as the output directory,
   * transform the relative path into the absolute path based on the current directory where the user runs REEF.
   *
   * @param outputDir path of the output directory given by the user
   * @param onLocal whether the path of the output directory given by the user is a local path
   * @return
   */
  private static String processOutputDir(final String outputDir, final boolean onLocal) {
    if (!onLocal) {
      return outputDir;
    }
    final File outputFile = new File(outputDir);
    return outputFile.getAbsolutePath();
  }

  /**
   * @return identifier.
   */
  public String getIdentifier() {
    return identifier;
  }

  /**
   * @return number of evaluators.
   */
  public int getEvalNum() {
    return evalNum;
  }

  /**
   * @return amount of memory space allocated to each evaluator.
   */
  public int getEvalSize() {
    return evalSize;
  }

  /**
   * @return path of the input directory.
   */
  public String getInputDir() {
    return inputDir;
  }

  /**
   * @return path of the output directory.
   */
  public boolean getOnLocal() {
    return onLocal;
  }

  /**
   * @return timeout on the job.
   */
  public int getTimeout() {
    return timeout;
  }
}
