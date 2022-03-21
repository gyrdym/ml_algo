import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KDTreeNode {
  KDTreeNode({this.value, this.index, this.left, this.right, this.samples});

  final Vector? value;
  final int? index;
  final KDTreeNode? left;
  final KDTreeNode? right;
  final Matrix? samples;

  bool get isLeaf => samples != null;

  bool testLeft(Vector sample) {
    return isLeaf || sample[index!] < value![index!];
  }
}
