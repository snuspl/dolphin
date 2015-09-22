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
package edu.snu.reef.dolphin.neuralnet.data;

import edu.snu.reef.dolphin.neuralnet.layers.LayerParameter;
import edu.snu.reef.dolphin.neuralnet.util.ValidationStats;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.util.Optional;

public final class NeuralNetParamServerData {
  private final boolean isValidationStatsPair;
  private final Optional<Pair<ValidationStats, ValidationStats>> validationStatsPair;
  private final Optional<LayerParameter[]> layerParameters;

  public NeuralNetParamServerData(final Pair<ValidationStats, ValidationStats> validationStatsPair) {
    this.isValidationStatsPair = true;
    this.validationStatsPair = Optional.of(validationStatsPair);
    this.layerParameters = Optional.empty();
  }

  public NeuralNetParamServerData(final LayerParameter[] layerParameters) {
    this.isValidationStatsPair = false;
    this.validationStatsPair = Optional.empty();
    this.layerParameters = Optional.of(layerParameters);
  }

  public boolean getIsValidationStatsPair() {
    return this.isValidationStatsPair;
  }

  public Optional<Pair<ValidationStats, ValidationStats>> getValidationStatsPair() {
    return this.validationStatsPair;
  }

  public Optional<LayerParameter[]> getLayerParameters() {
    return this.layerParameters;
  }
}
