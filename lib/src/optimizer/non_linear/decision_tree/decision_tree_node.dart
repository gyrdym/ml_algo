import 'package:ml_linalg/matrix.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.children, this.observations);

  final Iterable<DecisionTreeNode> children;
  final Matrix observations;
}
