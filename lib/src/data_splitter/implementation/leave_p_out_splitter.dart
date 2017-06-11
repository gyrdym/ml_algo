import 'package:dart_ml/src/data_splitter/interface/leave_p_out_splitter.dart';

class LeavePOutSplitterImpl implements LeavePOutSplitter {
  int _p = 2;

  void configure({int p = 2}) {
    if (p == 0) {
      throw new UnsupportedError('Value `$p` for parameter `p` is unsupported');
    }

    _p = p;
  }

  Iterable<Iterable<int>> split(int numberOfSamples) sync* {
    for (int u = 0; u < 1 << numberOfSamples; u++) {
      if (_count(u) == _p) {
        yield _generateCombination(u);
      }
    }
  }

  int _count(int u) {
    int n;
    for (n = 0; u > 0; ++n, u &= (u - 1)) {};
    return n;
  }

  Iterable<int> _generateCombination(int u) sync* {
    for (int n = 0; u > 0; ++n, u >>= 1) {
      if ((u & 1) > 0) {
        yield n;
      }
    }
  }
}
