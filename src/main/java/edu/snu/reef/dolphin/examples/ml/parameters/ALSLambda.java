package edu.snu.reef.dolphin.examples.ml.parameters;

import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

@NamedParameter(doc = "value for the weighted-regularization constant",
                short_name = "lambda",
                default_value = "0.05")
public final class ALSLambda implements Name<Double> {
}
