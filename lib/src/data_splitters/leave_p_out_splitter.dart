import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class LeavePOutSplitter {
  final int _p;

  LeavePOutSplitter({int p = 2}) : _p = p {
    if (p == 0 || p == 1) {
      throw new UnsupportedError('Value `$p` for parameter `p` is unsupported');
    }
  }

  List<List<int>> split(int length) {
    List<List<int>> folds = <List<int>>[];
    folds.add(new List<int>.generate(_p, (idx) => idx));
    int iteration = 0;
    do {
      int digit = _p - 1;
      int maxValForDigit = length - 1;
      do {
        List<int> fold = new List<int>.from(folds.last, growable: false);
        if (fold[digit] < maxValForDigit--) {
          ++fold[digit];
          folds.add(fold);
        }
      } while(digit-- != 0);
    } while (iteration++ != length);

    return folds;
  }
}
