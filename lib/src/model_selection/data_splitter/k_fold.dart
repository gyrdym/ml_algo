import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:xrange/zrange.dart';

class KFoldSplitter implements Splitter {
  KFoldSplitter(this._numberOfFolds) {
    if (_numberOfFolds == 0 || _numberOfFolds == 1) {
      throw RangeError(
          'Number of folds must be greater than 1 and less than number of samples');
    }
  }

  final int _numberOfFolds;

  @override
  Iterable<Iterable<int>> split(int numOfObservations) sync* {
    if (_numberOfFolds > numOfObservations) {
      throw RangeError.range(_numberOfFolds, 0, numOfObservations, null,
          'Number of folds must be less than number of samples!');
    }
    final remainder = numOfObservations % _numberOfFolds;
    final foldSize = numOfObservations ~/ _numberOfFolds;
    for (int i = 0, startIdx = 0, endIdx = 0; i < _numberOfFolds; i++) {
      endIdx = startIdx + foldSize + (i >= _numberOfFolds - remainder ? 1 : 0);
      yield ZRange.closedOpen(startIdx, endIdx).values();
      startIdx = endIdx;
    }
  }
}
