part of 'package:dart_ml/src/core/implementation.dart';

class _LeavePOutSplitterImpl implements Splitter {
  int _p = 2;

  _LeavePOutSplitterImpl(int p) {
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
