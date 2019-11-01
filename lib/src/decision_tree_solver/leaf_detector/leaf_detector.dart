import 'package:ml_linalg/matrix.dart';

abstract class DecisionTreeLeafDetector {
  bool isLeaf(Matrix sample, int targetIdx,
      Iterable<int> featureColumnIdxs, int treeDepth);
}
