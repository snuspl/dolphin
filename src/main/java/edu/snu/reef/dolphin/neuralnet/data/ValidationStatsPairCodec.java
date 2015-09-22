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

import edu.snu.reef.dolphin.neuralnet.util.ValidationStats;
import org.apache.reef.io.network.impl.StreamingCodec;
import org.apache.reef.io.network.util.Pair;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;

/**
 * Serialization codec for training/cross validation statistics.
 * Assumes training validation stats and cross validation stats are given as a {@code Pair}.
 */
public final class ValidationStatsPairCodec implements
    StreamingCodec<Pair<ValidationStats, ValidationStats>>, Codec<Pair<ValidationStats, ValidationStats>> {

  @Inject
  private ValidationStatsPairCodec() {
  }

  @Override
  public byte[] encode(final Pair<ValidationStats, ValidationStats> validationStatsPair) {
    try (final ByteArrayOutputStream bstream = new ByteArrayOutputStream(4 * Integer.SIZE);
         final DataOutputStream dstream = new DataOutputStream(bstream)) {
      encodeToStream(validationStatsPair, dstream);
      return bstream.toByteArray();

    } catch (final IOException e) {
      throw new RuntimeException("IOException during ValidationStatsPairCodec.encode()", e);
    }
  }

  @Override
  public void encodeToStream(final Pair<ValidationStats, ValidationStats> validationStatsPair,
                             final DataOutputStream dstream) {
    try {
      dstream.writeInt(validationStatsPair.getFirst().getTotalNum());
      dstream.writeInt(validationStatsPair.getFirst().getCorrectNum());
      dstream.writeInt(validationStatsPair.getSecond().getTotalNum());
      dstream.writeInt(validationStatsPair.getSecond().getCorrectNum());
    } catch (final IOException e) {
      throw new RuntimeException("IOException during ValidationStatsPairCodec.encodeToStream()", e);
    }
  }

  @Override
  public Pair<ValidationStats, ValidationStats> decode(final byte[] data) {
    try (final DataInputStream dstream = new DataInputStream(new ByteArrayInputStream(data))) {
      return decodeFromStream(dstream);

    } catch (final IOException e) {
      throw new RuntimeException("IOException during ValidationStatsPairCodec.decode()", e);
    }
  }

  @Override
  public Pair<ValidationStats, ValidationStats> decodeFromStream(final DataInputStream dstream) {
    try {
      final int firstTotalNum = dstream.readInt();
      final int firstCorrectNum = dstream.readInt();
      final int secondTotalNum = dstream.readInt();
      final int secondCorrectNum = dstream.readInt();
      return new Pair<>(new ValidationStats(firstTotalNum, firstCorrectNum),
          new ValidationStats(secondTotalNum, secondCorrectNum));
    } catch (final IOException e) {
      throw new RuntimeException("IOException during ValidationStatsPairCodec.decodeFromStream()", e);
    }
  }
}
