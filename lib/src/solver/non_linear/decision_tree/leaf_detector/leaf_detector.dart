import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class LeafDetector {
  bool isLeaf(Matrix sample, int targetIdx,
      Iterable<int> featureColumnIdxs, int treeDepth);
}
