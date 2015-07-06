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
package edu.snu.reef.dolphin.examples.ml.sub;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

/**
 * Codec for a vector.
 */
public class VectorCodec implements Codec<Vector> {
  /**
   * This class is instantiated by TANG.
   */
  @Inject
  public VectorCodec() {
    // Intentionally blank
  }

  @Override
  public Vector decode(final byte[] data) {
    try (final DataInputStream dis = new DataInputStream(new ByteArrayInputStream(data))) {
      final int size = dis.readInt();
      final Vector ret = new RandomAccessSparseVector(size);
      while (dis.available() > 0) {
        ret.setQuick(dis.readInt(), dis.readDouble());
      }
      return ret;
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }
  }

  @Override
  public byte[] encode(final Vector v) {
    final int size = v.size();
    try (final ByteArrayOutputStream baos = new ByteArrayOutputStream(size * Double.SIZE);
         final DataOutputStream dos = new DataOutputStream(baos)) {
      dos.writeInt(size);
      for (int i = 0; i < size; ++i) {
        final double value = v.get(i);
        if (value != 0) {
          dos.writeInt(i);
          dos.writeDouble(value);
        }
      }
      return baos.toByteArray();
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }
  }
}