import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider.dart';
import 'package:xrange/xrange.dart';

class KFoldIndicesProvider implements SplitIndicesProvider {
  KFoldIndicesProvider(this._numberOfFolds) {
    if (_numberOfFolds == 0 || _numberOfFolds == 1) {
      throw RangeError(
          'Number of folds must be greater than 1 and less than the number of '
              'samples');
    }
  }

  final int _numberOfFolds;

  @override
  Iterable<Iterable<int>> getIndices(int numOfObservations) sync* {
    if (_numberOfFolds > numOfObservations) {
      throw RangeError.range(_numberOfFolds, 0, numOfObservations, null,
          'Number of folds must be less than the number of samples');
    }
    final remainder = numOfObservations % _numberOfFolds;
    final foldSize = numOfObservations ~/ _numberOfFolds;
    for (var i = 0, startIdx = 0, endIdx = 0; i < _numberOfFolds; i++) {
      // if we reached the last fold of size [foldSize] then all the next folds
      // up to the last one have size of [foldSize] + 1
      endIdx = startIdx + foldSize + (i >= _numberOfFolds - remainder ? 1 : 0);
      yield integers(startIdx, endIdx);
      startIdx = endIdx;
    }
  }
}
