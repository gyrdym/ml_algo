import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';

class LeavePOutSplitter implements Splitter {
  LeavePOutSplitter([this._p = 2]) {
    if (_p == 0) {
      throw UnsupportedError('Value `$_p` for parameter `p` is unsupported');
    }
  }

  final int _p;

  @override
  Iterable<Iterable<int>> split(int numberOfSamples) sync* {
    for (int u = 0; u < 1 << numberOfSamples; u++) {
      if (_count(u) == _p) yield _generateCombination(u);
    }
  }

  int _count(int u) {
    int n;
    for (n = 0; u > 0; ++n, u &= (u - 1)) {}
    ;
    return n;
  }

  Iterable<int> _generateCombination(int u) sync* {
    for (int n = 0; u > 0; ++n, u >>= 1) {
      if ((u & 1) > 0) yield n;
    }
  }
}
