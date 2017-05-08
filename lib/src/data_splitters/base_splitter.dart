class SplitterHelper {
  const SplitterHelper();

  List<List<int>> split(int numberOfSamples, {int numberOfFolds = 5}) {
    if (numberOfFolds > numberOfSamples) {
      throw new RangeError.range(numberOfFolds, 0, numberOfSamples, null, 'Number of folds must be less than number of samples!');
    }

    int remainder = numberOfSamples % numberOfFolds;
    int size = (numberOfSamples / numberOfFolds).truncate();
    List<int> sizes = new List<int>.filled(numberOfFolds, 1).map((int el) => el * size).toList(growable: false);

    if (remainder > 0) {
      List<int> range = sizes.take(remainder).map((int el) => ++el).toList(growable: false);
      sizes.setRange(0, remainder, range);
    }

    int startIdx = 0;
    int endIdx = 0;
    List<List<int>> folds = [];

    for (int i = 0; i < sizes.length; i++) {
      endIdx = startIdx + sizes[i];
      folds.add(new List.from([startIdx, endIdx], growable: false));
      startIdx = endIdx;
    }

    return folds;
  }
}
