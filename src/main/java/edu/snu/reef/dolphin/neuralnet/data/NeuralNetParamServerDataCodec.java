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

import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

public final class NeuralNetParamServerDataCodec implements Codec<NeuralNetParamServerData> {
  private final LayerParameterArrayCodec layerParameterArrayCodec;
  private final ValidationStatsPairCodec validationStatsPairCodec;

  @Inject
  private NeuralNetParamServerDataCodec(final LayerParameterArrayCodec layerParameterArrayCodec,
                                        final ValidationStatsPairCodec validationStatsPairCodec) {
    this.layerParameterArrayCodec = layerParameterArrayCodec;
    this.validationStatsPairCodec = validationStatsPairCodec;
  }

  @Override
  public byte[] encode(final NeuralNetParamServerData neuralNetParamServerData) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {

      if (neuralNetParamServerData.getIsValidationStatsPair()) {
        dstream.writeBoolean(true);
        validationStatsPairCodec.encodeToStream(neuralNetParamServerData.getValidationStatsPair().get(), dstream);
      } else {
        dstream.writeBoolean(false);
        layerParameterArrayCodec.encodeToStream(neuralNetParamServerData.getLayerParameters().get(), dstream);
      }
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NeuralNetParamServerDataCodec.encode()", e);
    }
  }

  @Override
  public NeuralNetParamServerData decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      final boolean isValidationStatsPair = dstream.readBoolean();
      if (isValidationStatsPair) {
        return new NeuralNetParamServerData(validationStatsPairCodec.decodeFromStream(dstream));
      } else {
        return new NeuralNetParamServerData(layerParameterArrayCodec.decodeFromStream(dstream));
      }

    } catch (final IOException e) {
      throw new RuntimeException("IOException during NeuralNetParamServerDataCodec.decode()", e);
    }
  }
}
