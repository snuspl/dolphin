/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package edu.snu.reef.dolphin.examples.ml.algorithms.graph;

import edu.snu.reef.dolphin.core.DolphinConfiguration;
import edu.snu.reef.dolphin.core.DolphinLauncher;
import edu.snu.reef.dolphin.core.UserJobInfo;
import edu.snu.reef.dolphin.core.UserParameters;
import edu.snu.reef.dolphin.parameters.JobIdentifier;
import org.apache.commons.io.FileUtils;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.Assert;

import java.io.File;

/**
 * Launch the PageRank test.
 */
public final class PageRankTest {
  private static final String OUTPUT_PATH = "test_pagerank";

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
   * Run PageRank test.
   */
  @Test
  public void testPageRank() throws Exception {
    final String[] args = {
        "-convThr", "0.01", 
        "-maxIter", "10", 
        "-dampingFactor", "0.85", 
        "-local", "true", 
        "-split", "1",
        "-input", ClassLoader.getSystemResource("data").getPath() + "/pagerank", 
        "-output", OUTPUT_PATH
    };

    DolphinLauncher.run(
        Configurations.merge(
            DolphinConfiguration.getConfiguration(args, PageRankParameters.getCommandLine()),
            Tang.Factory.getTang().newConfigurationBuilder()
                .bindNamedParameter(JobIdentifier.class, "PageRank")
                .bindImplementation(UserJobInfo.class, PageRankJobInfo.class)
                .bindImplementation(UserParameters.class, PageRankParameters.class)
                .build()
        )
    );

    final File expected = new File(ClassLoader.getSystemResource("result").getPath() + "/pagerank");
    final File actual = new File(OUTPUT_PATH + "/rank/CtrlTask-0");

    Assert.assertTrue(FileUtils.contentEquals(expected, actual));
  }
}
