/**
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
