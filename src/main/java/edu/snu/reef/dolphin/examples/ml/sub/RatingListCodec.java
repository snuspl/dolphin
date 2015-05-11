package edu.snu.reef.dolphin.examples.ml.sub;

import edu.snu.reef.dolphin.examples.ml.data.Rating;
import org.apache.reef.io.serialization.Codec;

import javax.inject.Inject;
import java.io.*;
import java.util.LinkedList;
import java.util.List;

public final class RatingListCodec implements Codec<List<Rating>> {

  @Inject
  public RatingListCodec() {
  }

  @Override
  public final byte[] encode(final List<Rating> list) {
    final ByteArrayOutputStream baos =
        new ByteArrayOutputStream(Integer.SIZE +
                                  Integer.SIZE * 2 * list.size() +
                                  Double.SIZE * list.size());

    try (final DataOutputStream daos = new DataOutputStream(baos)) {
      daos.writeInt(list.size());
      for (final Rating rating : list) {
        daos.writeInt(rating.getUserIndex());
        daos.writeInt(rating.getItemIndex());
        daos.writeDouble(rating.getRatingScore());
      }
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return baos.toByteArray();
  }

  @Override
  public final List<Rating> decode(final byte[] data) {
    final ByteArrayInputStream bais = new ByteArrayInputStream(data);
    final List<Rating> resultList = new LinkedList<>();

    try (final DataInputStream dais = new DataInputStream(bais)) {
      final int listSize = dais.readInt();
      for (int i = 0; i < listSize; i++) {
        final int userIndex = dais.readInt();
        final int itemIndex = dais.readInt();
        final double ratingScore = dais.readDouble();
        resultList.add(new Rating(userIndex, itemIndex, ratingScore));
      }
    } catch (final IOException e) {
      throw new RuntimeException(e.getCause());
    }

    return resultList;
  }
}
