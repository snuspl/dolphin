package edu.snu.reef.dolphin.examples.ml.sub;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;
import java.util.Map;
import java.util.TreeMap;

public final class MapOfIntVecCodec implements Codec<Map<Integer, Vector>> {

  @Inject
  public MapOfIntVecCodec() {
  }

  @Override
  public final byte[] encode(final Map<Integer, Vector> map) {
    int vectorSizeSum = 0;
    for (final Vector vector : map.values()) {
      vectorSizeSum += vector.size();
    }

    final ByteArrayOutputStream baos =
        new ByteArrayOutputStream(Integer.SIZE
                                  + Integer.SIZE * 2 * map.size()
                                  + Double.SIZE * vectorSizeSum);
    try (final DataOutputStream daos = new DataOutputStream(baos)) {
      daos.writeInt(map.size());

      for (final int index : map.keySet()) {
        final Vector vector = map.get(index);
        daos.writeInt(index);
        daos.writeInt(vector.size());
        for (int vecIndex = 0; vecIndex < vector.size(); vecIndex++) {
          daos.writeDouble(vector.get(vecIndex));
        }
      }

    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return baos.toByteArray();
  }

  @Override
  public final Map<Integer, Vector> decode(final byte[] data) {
    final Map<Integer, Vector> retMap = new TreeMap<>();

    final ByteArrayInputStream bais = new ByteArrayInputStream(data);
    try (final DataInputStream dais = new DataInputStream(bais)) {
      final int mapSize = dais.readInt();

      for (int count = 0; count < mapSize; count++) {
        final int index = dais.readInt();
        final int vecSize = dais.readInt();
        final Vector vector = new DenseVector(vecSize);
        for (int vecIndex = 0; vecIndex < vecSize; vecIndex++) {
          vector.set(vecIndex, dais.readDouble());
        }
        retMap.put(index, vector);
      }

    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return retMap;
  }
}
