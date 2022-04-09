enum TreeAssessorType {
  /// According to this assessor type, the decision tree algorithm makes a
  /// decision on how to split a subset of data based on a major class.
  majority,

  /// According to this assessor type, the decision tree algorithm makes a
  /// decision on how to split a subset of data based on the [Gini index](https://en.wikipedia.org/wiki/Gini_coefficient)
  gini,
}
