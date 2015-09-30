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

import edu.snu.dolphin.dnn.layers.LayerParameter;
import org.apache.reef.io.network.impl.StreamingCodec;
import org.apache.reef.io.serialization.Codec;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.inject.Inject;
import java.io.*;

/**
 * Serialization codec for neural network layer parameters (weights and biases).
 * Internally uses {@link NDArrayCodec}.
 */
public final class LayerParameterArrayCodec implements StreamingCodec<LayerParameter[]>, Codec<LayerParameter[]> {

  private final NDArrayCodec ndArrayCodec;

  @Inject
  private LayerParameterArrayCodec(final NDArrayCodec ndArrayCodec) {
    this.ndArrayCodec = ndArrayCodec;
  }

  @Override
  public byte[] encode(final LayerParameter[] layerParameters) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {
      encodeToStream(layerParameters, dstream);
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final LayerParameter[] layerParameters, final DataOutputStream dstream) {
    try {
      dstream.writeInt(layerParameters.length);
      for (final LayerParameter layerParameter : layerParameters) {
        ndArrayCodec.encodeToStream(layerParameter.getWeightParam(), dstream);
        ndArrayCodec.encodeToStream(layerParameter.getBiasParam(), dstream);
      }

    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayCodec.encodeToStream()", e);
    }
  }

  @Override
  public LayerParameter[] decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayCodec.decode()", e);
    }
  }

  @Override
  public LayerParameter[] decodeFromStream(final DataInputStream dstream) {
    try {
      final LayerParameter[] layerParameters = new LayerParameter[dstream.readInt()];
      for (int index = 0; index < layerParameters.length; index++) {
        final INDArray weightParam = ndArrayCodec.decodeFromStream(dstream);
        final INDArray biasParam = ndArrayCodec.decodeFromStream(dstream);
        layerParameters[index] = LayerParameter.newBuilder()
            .setWeightParam(weightParam)
            .setBiasParam(biasParam)
            .build();
      }

      return layerParameters;

    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayCodec.decodeFromStream()", e);
    }
  }
}
