package edu.snu.reef.dolphin.examples.ml.parameters;

import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

@NamedParameter(doc = "Number of features to model users and vectors",
                short_name = "featureNum",
                default_value = "10")
public final class FeatureNum implements Name<Integer> {
}
