package edu.snu.reef.flexion.groupcomm.subs;

import com.microsoft.reef.io.network.group.operators.Reduce;

import javax.inject.Inject;
import java.util.ArrayList;
import java.util.List;

public final class DataReduceFunction implements Reduce.ReduceFunction<Integer> {

  @Inject
  public DataReduceFunction() {
  }

  @Override
  public final Integer apply(Iterable<Integer> dataList) {
    Integer sum = 0;
    Integer count = 0;
    for (final Integer data : dataList) {
      sum += data;
      count++;
    }

    return sum / count;
  }
}