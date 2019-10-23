void validateClassLabelList(Iterable<num> classLabels) {
  if (classLabels.isEmpty) {
    throw Exception('Empty class label list provided');
  }
}
