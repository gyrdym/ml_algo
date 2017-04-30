List<List<int>> getKFoldIndices(int numberOfSamples,  {int fold: 5}) {
  int remainder = numberOfSamples % fold;
  int size = (numberOfSamples / fold).truncate();
  List<int> sizes = new List<int>.filled(fold, 1).map((int el) => el * size).toList(growable: false);

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