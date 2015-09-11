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
import org.apache.reef.io.network.impl.StreamingCodec;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Serialization codec for a list of neural network parameters arrays.
 * Internally uses {@link LayerParameterArrayCodec}.
 */
public final class LayerParameterArrayListCodec
    implements StreamingCodec<List<LayerParameter[]>>, Codec<List<LayerParameter[]>> {

  private final LayerParameterArrayCodec layerParameterArrayCodec;

  @Inject
  private LayerParameterArrayListCodec(final LayerParameterArrayCodec layerParameterArrayCodec) {
    this.layerParameterArrayCodec = layerParameterArrayCodec;
  }

  @Override
  public byte[] encode(final List<LayerParameter[]> layerParametersList) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream();
         final DataOutputStream dstream = new DataOutputStream(bstream)) {
      encodeToStream(layerParametersList, dstream);
      return bstream.toByteArray();
    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayListCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final List<LayerParameter[]> layerParametersList,
                             final DataOutputStream dstream) {
    try {
      dstream.writeInt(layerParametersList.size());
      for (final LayerParameter[] layerParameters : layerParametersList) {
        layerParameterArrayCodec.encodeToStream(layerParameters, dstream);
      }
    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayListCodec.encodeToStream()", e);
    }
  }

  @Override
  public List<LayerParameter[]> decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);
    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayList.decode()", e);
    }
  }

  @Override
  public List<LayerParameter[]> decodeFromStream(final DataInputStream dstream) {
    try {
      final int size = dstream.readInt();
      final List<LayerParameter[]> retList = new ArrayList<>(size);
      for (int i = 0; i < size; ++i) {
        retList.add(layerParameterArrayCodec.decodeFromStream(dstream));
      }
      return retList;
    } catch (final IOException e) {
      throw new RuntimeException("IOException during LayerParameterArrayListCodec.decodeToStream()", e);
    }
  }
}
