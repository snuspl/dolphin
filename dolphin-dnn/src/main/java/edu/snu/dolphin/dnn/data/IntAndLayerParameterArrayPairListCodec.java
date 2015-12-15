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
import java.util.ArrayList;
import java.util.List;

/**
 * Serialization codec for a list of pairs of an integer and a layer parameter array.
 * Internally uses {@link IntAndLayerParameterArrayPairCodec}.
 */
public final class IntAndLayerParameterArrayPairListCodec
    implements StreamingCodec<List<Pair<Integer, LayerParameter[]>>>, Codec<List<Pair<Integer, LayerParameter[]>>> {

  private final IntAndLayerParameterArrayPairCodec intAndLayerParameterArrayPairCodec;

  @Inject
  private IntAndLayerParameterArrayPairListCodec(
      final IntAndLayerParameterArrayPairCodec intAndLayerParameterArrayPairCodec) {
    this.intAndLayerParameterArrayPairCodec = intAndLayerParameterArrayPairCodec;
  }

  @Override
  public byte[] encode(final List<Pair<Integer, LayerParameter[]>> intAndLayerParametersPairList) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {
      encodeToStream(intAndLayerParametersPairList, dstream);
      return bstream.toByteArray();
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairListCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final List<Pair<Integer, LayerParameter[]>> intAndLayerParameterPairList,
                             final DataOutputStream dstream) {
    try {
      dstream.writeInt(intAndLayerParameterPairList.size());
      for (final Pair<Integer, LayerParameter[]> intAndLayerParameterPair : intAndLayerParameterPairList) {
        intAndLayerParameterArrayPairCodec.encodeToStream(intAndLayerParameterPair, dstream);
      }
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairListCodec.encodeToStream()", e);
    }
  }

  @Override
  public List<Pair<Integer, LayerParameter[]>> decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairListCodec.decode()", e);
    }
  }

  @Override
  public List<Pair<Integer, LayerParameter[]>> decodeFromStream(final DataInputStream dstream) {
    try {
      final int size = dstream.readInt();
      final List<Pair<Integer, LayerParameter[]>> ret = new ArrayList<>(size);
      for (int i = 0; i < size; ++i) {
        ret.add(intAndLayerParameterArrayPairCodec.decodeFromStream(dstream));
      }
      return ret;
    } catch (final IOException e) {
      throw new RuntimeException("IOException during IntAndLayerParameterArrayPairListCodec.decodeFromStream()", e);
    }
  }
}
