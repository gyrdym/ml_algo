import 'package:ml_linalg/matrix.dart';

abstract class TreeLeafDetector {
  bool isLeaf(Matrix sample, int targetIdx,
      Iterable<int> featureColumnIdxs, int treeDepth);
}
