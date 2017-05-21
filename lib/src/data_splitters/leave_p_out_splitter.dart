import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class LeavePOutSplitter implements SplitterInterface {
  final int _p;

  LeavePOutSplitter({int p = 2}) : _p = p {
    if (p == 0 || p == 1) {
      throw new UnsupportedError('Value `$p` for parameter `p` is unsupported');
    }
  }

  Iterable<Iterable<int>> split(int numberOfSamples) sync* {
    for (int u = 0; u < 1 << numberOfSamples; u++) {
      if (_bitCount(u) == _p) {
        yield _generatePart(u);
      }
    }
  }

  int _bitCount(int u) {
    int n;
    for (n = 0; u > 0; ++n, u &= (u - 1)) {};
    return n;
  }

  Iterable<int> _generatePart(int u) sync* {
    for (int n = 0; u > 0; ++n, u >>= 1) {
      if ((u & 1) > 0) {
        yield n;
      }
    }
  }
}
