/**
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
package edu.snu.reef.flexion.examples.ml.algorithms.regression;

import edu.snu.reef.flexion.core.DataParser;
import edu.snu.reef.flexion.core.ParseException;
import edu.snu.reef.flexion.core.UserComputeTask;
import edu.snu.reef.flexion.examples.ml.data.LinearModel;
import edu.snu.reef.flexion.examples.ml.data.Row;
import edu.snu.reef.flexion.examples.ml.data.LinearRegSummary;
import edu.snu.reef.flexion.examples.ml.loss.Loss;
import edu.snu.reef.flexion.examples.ml.parameters.StepSize;
import edu.snu.reef.flexion.examples.ml.regularization.Regularization;
import edu.snu.reef.flexion.groupcomm.interfaces.DataBroadcastReceiver;
import edu.snu.reef.flexion.groupcomm.interfaces.DataReduceSender;
import org.apache.mahout.math.Vector;
import org.apache.reef.tang.annotations.Parameter;

import javax.inject.Inject;
import java.util.List;

public class LinearRegCmpTask extends UserComputeTask
    implements DataReduceSender<LinearRegSummary>, DataBroadcastReceiver<LinearModel> {
  private double stepSize;
  private final Loss loss;
  private final Regularization regularization;
  private DataParser<List<Row>> dataParser;
  private List<Row> rows;
  private LinearModel model;
  private double lossSum = 0;

  @Inject
  public LinearRegCmpTask(@Parameter(StepSize.class) final double stepSize,
                          final Loss loss,
                          final Regularization regularization,
                          DataParser<List<Row>> dataParser) {
    this.stepSize = stepSize;
    this.loss = loss;
    this.regularization = regularization;
    this.dataParser = dataParser;
  }

  @Override
  public void initialize() throws ParseException {
    rows = dataParser.get();
  }

  @Override
  public final void run(int iteration) {

    // measure loss
    lossSum = 0;
    for (final Row row : rows) {
      final double output = row.getOutput();
      final double predict = model.predict(row.getFeature());
      lossSum += loss.loss(predict, output);
    }

    // optimize
    for (final Row row : rows) {
      final double output = row.getOutput();
      final Vector input = row.getFeature();
      final Vector gradient = loss.gradient(input, model.predict(input), output).plus(regularization.gradient(model));
      model.setParameters(model.getParameters().minus(gradient.times(stepSize)));
    }
  }

  @Override
  public final void receiveBroadcastData(int iteration, LinearModel model) {
    this.model = model;
  }

  @Override
  public LinearRegSummary sendReduceData(int iteration) {
    return new LinearRegSummary(this.model, 1, this.lossSum);
  }
}