import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/from_tree_nodes_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/tree_nodes_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/_helper/get_tree_node_splitting_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/from_tree_node_splitting_predicate_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';
import 'package:ml_linalg/vector.dart';

part 'tree_node.g.dart';

@JsonSerializable()
class TreeNode {
  TreeNode(
    this.predicateType,
    this.splittingValue,
    this.splittingIndex,
    this.children,
    this.label, [
    this.level = 0,
  ]);

  factory TreeNode.fromJson(Map<String, dynamic> json) =>
      _$TreeNodeFromJson(json);

  Map<String, dynamic> toJson() => _$TreeNodeToJson(this);

  @JsonKey(
    name: childrenJsonKey,
    toJson: treeNodesToJson,
    fromJson: fromTreeNodesJson,
  )
  final List<TreeNode>? children;

  @JsonKey(name: labelJsonKey)
  final TreeLeafLabel? label;

  @JsonKey(
    name: predicateTypeJsonKey,
    toJson: splittingPredicateTypeToJson,
    fromJson: fromSplittingPredicateTypeJson,
  )
  final TreeNodeSplittingPredicateType? predicateType;

  @JsonKey(name: splittingValueJsonKey)
  final num? splittingValue;

  @JsonKey(name: splittingIndexJsonKey)
  final int? splittingIndex;

  @JsonKey(name: levelJsonKey)
  final int level;

  bool get isLeaf => children == null || children!.isEmpty;

  bool get isRoot =>
      predicateType == null || splittingIndex == null || splittingValue == null;

  bool get isFake =>
      predicateType == null &&
      splittingIndex == null &&
      splittingValue == null &&
      label == null &&
      children == null;

  bool isSamplePassed(Vector sample) {
    if (isRoot) {
      return true;
    }

    final isSamplePassedFn =
        getTreeNodeSplittingPredicateByType(predicateType!);

    return isSamplePassedFn(
      sample,
      splittingIndex!,
      splittingValue!,
    );
  }

  /// Returns a map where a key is a tree level number, and the value is a
  /// number of nodes on the level
  Map<int, int> get shape {
    final _shape = <int, int>{};

    _collectShape(this, _shape, 0);

    return _shape;
  }

  void _collectShape(TreeNode node, Map<int, int> shape, int level) {
    final children = node.children;
    final childCount = children?.length ?? 0;

    shape.update(level, (count) => count + childCount,
        ifAbsent: () => childCount);

    children?.forEach((child) {
      _collectShape(child, shape, level + 1);
    });
  }
}
