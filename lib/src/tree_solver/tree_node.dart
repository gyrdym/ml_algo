import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label.dart';
import 'package:ml_linalg/vector.dart';

typedef TestSamplePredicate = bool Function(Vector sample);

class TreeNode with SerializableMixin {
  TreeNode(
      this.testSample,
      this.splittingValue,
      this.splittingIdx,
      this.children,
      this.label,
      [
        this.level = 0,
      ]
  );

  final List<TreeNode> children;
  final TreeLeafLabel label;
  final TestSamplePredicate testSample;
  final num splittingValue;
  final int splittingIdx;
  final int level;

  bool get isLeaf => children == null || children.isEmpty;

  @override
  Map<String, dynamic> serialize() => <String, dynamic>{
    'splittingValue': splittingValue,
    'splittingIdx': splittingIdx,
    'level': level,
    'label': label?.serialize(),
    'children': children?.map((node) => node.serialize())?.toList(),
  };
}
