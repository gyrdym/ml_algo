List<List<int>> getKFoldIndices(int numberOfSamples,  {int folds: 5}) {
  if (folds > numberOfSamples) {
    throw new RangeError.range(folds, 0, numberOfSamples, null, 'Number of folds must be less than number of samples!');
  }

  int remainder = numberOfSamples % folds;
  int size = (numberOfSamples / folds).truncate();
  List<int> sizes = new List<int>.filled(folds, 1).map((int el) => el * size).toList(growable: false);

  if (remainder > 0) {
    List<int> range = sizes.take(remainder).map((int el) => ++el).toList(growable: false);
    sizes.setRange(0, remainder, range);
  }

  int startIdx = 0;
  List<List<int>> indices = [];

  for (int i = 0; i < sizes.length; i++) {
    int endIdx = startIdx + sizes[i];
    indices.add([startIdx, endIdx]);
    startIdx = endIdx;
  }

  return indices;
}