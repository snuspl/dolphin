package edu.snu.reef.dolphin.examples.ml.data;

import edu.snu.reef.dolphin.core.DataParser;
import edu.snu.reef.dolphin.core.ParseException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.reef.io.data.loading.api.DataSet;
import org.apache.reef.io.network.util.Pair;

import javax.inject.Inject;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public final class RecommendationDataParser implements DataParser<List<Rating>> {
  private final static Logger LOG = Logger.getLogger(RecommendationDataParser.class.getName());

  private final DataSet<LongWritable, Text> dataSet;
  private List<Rating> result;
  private ParseException parseException;

  @Inject
  public RecommendationDataParser(final DataSet<LongWritable, Text> dataSet) {
    this.dataSet = dataSet;
  }

  @Override
  public final List<Rating> get() throws ParseException {
    if (result == null) {
      parse();
    }

    if (parseException != null) {
      throw parseException;
    }

    return result;
  }

  @Override
  public final void parse() {
    LOG.log(Level.INFO, "Parsing.");
    result = new LinkedList<>();

    for (final Pair<LongWritable, Text> keyValue : dataSet) {
      final String text = keyValue.second.toString().trim();
      if (text.startsWith("#") || text.isEmpty()) {
        continue;
      }

      final String[] split = text.split("\\s+");
      if (split.length != 3) {
        parseException = new ParseException(
            "Parse failed: format must be 'userIndex itemIndex ratingScore'");
        return;
      }

      try {
        final int userIndex = Integer.valueOf(split[0]);
        final int itemIndex = Integer.valueOf(split[1]);
        final double ratingScore = Double.valueOf(split[2]);
        result.add(new Rating(userIndex, itemIndex, ratingScore));

      } catch (final NumberFormatException e) {
        parseException = new ParseException(
            "Parse failed: indices should be INTEGER, rating should be DOUBLE");
        return;
      }
    }

    LOG.log(Level.INFO, "my  result is: {0}", result);
  }
}
