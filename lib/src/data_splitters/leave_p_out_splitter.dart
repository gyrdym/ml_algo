import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class LeavePOutSplitter implements SplitterInterface {
  final int _p;

  LeavePOutSplitter({int p = 1}) : _p = p;

  List<List<int>> split(int samplesLength) {
    List<List<int>> folds = <List<int>>[];
    List<int> fold;
    int counter;

    void _initNewFold(int start) {
      fold = new List<int>(_p);
      fold[0] = start;
      counter = 1;
    }

    for (int startIndex = 0; startIndex < samplesLength; startIndex++) {
      _initNewFold(startIndex);

      for (int idx = 0; idx < samplesLength; idx++) {
        if (idx <= startIndex) {
          continue;
        }

        fold[counter++] = idx;

        if (counter == _p) {
          folds.add(fold);
          _initNewFold(startIndex);
        }
      }
    }

    return folds;
  }
}
