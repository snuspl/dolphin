package edu.snu.reef.dolphin.examples.ml.data;

public final class Rating {
  private final int userIndex;
  private final int itemIndex;
  private final double ratingScore;
  
  public Rating(final int userIndex, final int itemIndex, final double ratingScore) {
    this.userIndex = userIndex;
    this.itemIndex = itemIndex;
    this.ratingScore = ratingScore;
  }
  
  public final int getUserIndex() {
    return userIndex;
  }
  
  public final int getItemIndex() {
    return itemIndex;
  }
  
  public final double getRatingScore() {
    return ratingScore;
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("Rating(")
        .append(userIndex)
        .append(", ")
        .append(itemIndex)
        .append(", ")
        .append(ratingScore)
        .append(")");
    return sb.toString();
  }
}
