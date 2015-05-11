package edu.snu.reef.dolphin.examples.ml.key;

import edu.snu.reef.dolphin.core.Key;
import edu.snu.reef.dolphin.examples.ml.data.Rating;
import org.apache.mahout.math.Vector;
import org.apache.reef.io.network.util.Pair;

import java.util.Collection;
import java.util.List;
import java.util.Map;

public final class ItemRatings implements Key<Map<Integer, Pair<Collection<Integer>, Vector>>> {
}
