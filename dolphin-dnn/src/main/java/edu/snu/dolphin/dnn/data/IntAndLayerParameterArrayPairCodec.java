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
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

/**
 * Serialization codec for a pair of an integer and a layer parameter array.
 * Internally uses {@link LayerParameterArrayCodec}.
 */
public final class IntAndLayerParameterArrayPairCodec
    implements StreamingCodec<Pair<Integer, LayerParameter[]>>, Codec<Pair<Integer, LayerParameter[]>> {

  private final LayerParameterArrayCodec layerParameterArrayCodec;

  @Inject
  private IntAndLayerParameterArrayPairCodec(final LayerParameterArrayCodec layerParameterArrayCodec) {
    this.layerParameterArrayCodec = layerParameterArrayCodec;
  }

  @Override
  public byte[] encode(final Pair<Integer, LayerParameter[]> intAndLayerParametersPair) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {
      encodeToStream(intAndLayerParametersPair, dstream);
      return bstream.toByteArray();
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final Pair<Integer, LayerParameter[]> intAndLayerParametersPair,
                             final DataOutputStream dstream) {
    try {
      dstream.writeInt(intAndLayerParametersPair.getFirst());
      layerParameterArrayCodec.encodeToStream(intAndLayerParametersPair.getSecond(), dstream);
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairCodec.encodeToStream()", e);
    }
  }

  @Override
  public Pair<Integer, LayerParameter[]> decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairCodec.decode()", e);
    }
  }

  @Override
  public Pair<Integer, LayerParameter[]> decodeFromStream(final DataInputStream dstream) {
    try {
      final int firstValue = dstream.readInt();
      return new Pair<>(firstValue, layerParameterArrayCodec.decodeFromStream(dstream));
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairCodec.decodeFromStream()", e);
    }
  }
}
