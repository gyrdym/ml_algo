import 'package:dart_ml/src/data_splitter/splitter.dart';

class KFoldSplitter implements Splitter {
  final int _numberOfFolds;

  KFoldSplitter(this._numberOfFolds);

  @override
  Iterable<Iterable<int>> split(int numberOfSamples) sync* {
    if (_numberOfFolds == 0 || _numberOfFolds == 1) {
      throw new RangeError.range(_numberOfFolds, 2, numberOfSamples, null,
        'Number of folds must be greater than 1 and less than number of samples');
    }

    if (_numberOfFolds > numberOfSamples) {
      throw new RangeError.range(_numberOfFolds, 0, numberOfSamples, null, 'Number of folds must be less than number of samples!');
    }

    final remainder = numberOfSamples % _numberOfFolds;
    final size = (numberOfSamples / _numberOfFolds).truncate();
    final sizes = new List<int>.filled(_numberOfFolds, 1).map((int el) => el * size).toList(growable: false);

    if (remainder > 0) {
      final range = sizes.take(remainder).map((int el) => ++el).toList(growable: false);
      sizes.setRange(0, remainder, range);
    }

    int startIdx = 0;
    int endIdx = 0;

    for (int i = 0; i < sizes.length; i++) {
      endIdx = startIdx + sizes[i];
      yield _range(startIdx, endIdx);
      startIdx = endIdx;
    }
  }

  Iterable<int> _range(int start, int end) sync* {
    for (int i = start ; i < end; i++) {yield i;}
  }
}
