import 'package:ml_linalg/matrix.dart';

abstract class LeafDetector {
  bool isLeaf(Matrix observations, int nodesCount);
}
