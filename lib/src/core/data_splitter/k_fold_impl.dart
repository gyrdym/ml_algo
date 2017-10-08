part of 'package:dart_ml/src/core/implementation.dart';

class _KFoldSplitterImpl implements KFoldSplitter {
  int _numberOfFolds = 5;

  @override
  void configure({int numberOfFolds = 5}) {
    _numberOfFolds = numberOfFolds;
  }

  @override
  Iterable<Iterable<int>> split(int numberOfSamples) sync* {
    if (_numberOfFolds > numberOfSamples) {
      throw new RangeError.range(_numberOfFolds, 0, numberOfSamples, null, 'Number of folds must be less than number of samples!');
    }

    int remainder = numberOfSamples % _numberOfFolds;
    int size = (numberOfSamples / _numberOfFolds).truncate();
    List<int> sizes = new List<int>.filled(_numberOfFolds, 1).map((int el) => el * size).toList(growable: false);

    if (remainder > 0) {
      List<int> range = sizes.take(remainder).map((int el) => ++el).toList(growable: false);
      sizes.setRange(0, remainder, range);
    }

    int startIdx = 0;
    int endIdx = 0;

    for (int i = 0; i < sizes.length; i++) {
      endIdx = startIdx + sizes[i];
      yield _generateRange(startIdx, endIdx);
      startIdx = endIdx;
    }
  }

  Iterable<int> _generateRange(int start, int end) sync* {
    for (int i = start ; i < end; i++) {
      yield i;
    }
  }
}
