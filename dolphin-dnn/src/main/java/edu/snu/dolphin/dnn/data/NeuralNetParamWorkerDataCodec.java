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
package edu.snu.dolphin.dnn.data;

import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

public final class NeuralNetParamWorkerDataCodec implements Codec<NeuralNetParamWorkerData> {

  private final LayerParameterArrayListCodec layerParameterArrayListCodec;
  private final ValidationStatsPairCodec validationStatsPairCodec;

  @Inject
  private NeuralNetParamWorkerDataCodec(final LayerParameterArrayListCodec layerParameterArrayListCodec,
                                        final ValidationStatsPairCodec validationStatsPairCodec) {
    this.layerParameterArrayListCodec = layerParameterArrayListCodec;
    this.validationStatsPairCodec = validationStatsPairCodec;
  }

  @Override
  public byte[] encode(final NeuralNetParamWorkerData neuralNetParamWorkerData) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {

      if (neuralNetParamWorkerData.getIsValidationStatsPair()) {
        dstream.writeBoolean(true);
        validationStatsPairCodec.encodeToStream(neuralNetParamWorkerData.getValidationStatsPair().get(), dstream);
      } else {
        dstream.writeBoolean(false);
        layerParameterArrayListCodec.encodeToStream(neuralNetParamWorkerData.getLayerParametersList().get(), dstream);
      }
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NeuralNetParamWorkerDataCodec.encode()", e);
    }
  }

  @Override
  public NeuralNetParamWorkerData decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      final boolean isValidationStatsPair = dstream.readBoolean();
      if (isValidationStatsPair) {
        return new NeuralNetParamWorkerData(validationStatsPairCodec.decodeFromStream(dstream));
      } else {
        return new NeuralNetParamWorkerData(layerParameterArrayListCodec.decodeFromStream(dstream));
      }

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NeuralNetParamWorkerDataCodec.decode()", e);
    }
  }
}
