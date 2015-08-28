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
package edu.snu.reef.dolphin.examples.ml.algorithms.regression;

import edu.snu.reef.dolphin.core.DolphinConfiguration;
import edu.snu.reef.dolphin.core.DolphinLauncher;
import edu.snu.reef.dolphin.core.UserJobInfo;
import edu.snu.reef.dolphin.core.UserParameters;
import edu.snu.reef.dolphin.parameters.JobIdentifier;
import org.apache.commons.io.FileUtils;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

/**
 * Launch the LinearRegTest test.
 */
public final class LinearRegTest {
  private static final String OUTPUT_PATH = "target/test-linear-reg";

  /**
   * Set up the test environment.
   */
  @Before
  public void setUp() throws Exception {
    FileUtils.deleteDirectory(new File(OUTPUT_PATH));
  }

  /**
   * Tear down the test environment.
   */
  @After
  public void tearDown() throws Exception {
  }

  /**
   * Run LinearReg test.
   */
  @Test
  public void testLinearReg() throws Exception {
    final String[] args = {
        "-dim", "3",
        "-maxIter", "20",
        "-stepSize", "0.001",
        "-lambda", "0.1",
        "-local", "true",
        "-split", "4",
        "-input", ClassLoader.getSystemResource("data").getPath() + "/regression",
        "-output", OUTPUT_PATH,
        "-maxNumEvalLocal", "5"
    };

    DolphinLauncher.run(
        Configurations.merge(
            DolphinConfiguration.getConfiguration(args, LinearRegParameters.getCommandLine()),
            Tang.Factory.getTang().newConfigurationBuilder()
                .bindNamedParameter(JobIdentifier.class, "Linear Regression")
                .bindImplementation(UserJobInfo.class, LinearRegJobInfo.class)
                .bindImplementation(UserParameters.class, LinearRegParameters.class)
                .build()
        )
    );

    final File expectedModel = new File(
        ClassLoader.getSystemResource("result").getPath() + "/linearreg_model");
    final File actualModel = new File(OUTPUT_PATH + "/model/CtrlTask-0");
    Assert.assertTrue(FileUtils.contentEquals(expectedModel, actualModel));

    final File expectedAccuracy = new File(
        ClassLoader.getSystemResource("result").getPath() + "/linearreg_loss");
    final File actualAccuracy = new File(OUTPUT_PATH + "/loss/CtrlTask-0");
    Assert.assertTrue(FileUtils.contentEquals(expectedAccuracy, actualAccuracy));
  }
}
