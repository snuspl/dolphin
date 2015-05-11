package edu.snu.reef.dolphin.examples.ml.algorithms.recommendation;

import edu.snu.reef.dolphin.core.DolphinConfiguration;
import edu.snu.reef.dolphin.core.DolphinLauncher;
import edu.snu.reef.dolphin.core.UserJobInfo;
import edu.snu.reef.dolphin.core.UserParameters;
import edu.snu.reef.dolphin.parameters.JobIdentifier;
import org.apache.reef.tang.Configurations;
import org.apache.reef.tang.Tang;

public class ALSREEF {
  public final static void main(String[] args) throws Exception {
    DolphinLauncher.run(Configurations.merge(
            DolphinConfiguration.CONF(args, ALSParameters.getCommandLine()),
            Tang.Factory.getTang().newConfigurationBuilder()
                .bindNamedParameter(JobIdentifier.class, "Alternative Least Squares")
                .bindImplementation(UserJobInfo.class, ALSJobInfo.class)
                .bindImplementation(UserParameters.class, ALSParameters.class)
                .build()));
  }
}
