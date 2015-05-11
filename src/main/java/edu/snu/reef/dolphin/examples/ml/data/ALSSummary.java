package edu.snu.reef.dolphin.examples.ml.data;

import org.apache.mahout.math.Matrix;

public final class ALSSummary {
  private final Matrix matrix;
  private final UserItem userItem;

  public ALSSummary(final Matrix matrix, final UserItem userItem) {
    this.matrix = matrix;
    this.userItem = userItem;
  }

  public final Matrix getMatrix() {
    return matrix;
  }

  public final UserItem getUserItem() {
    return userItem;
  }

  public enum UserItem {
    USER, ITEM
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("ALSSummary(")
        .append(userItem == UserItem.USER ? "USER" : "ITEM")
        .append(", ")
        .append(System.getProperty("line.separator"))
        .append(matrix.toString())
        .append(")");
    return sb.toString();
  }
}
