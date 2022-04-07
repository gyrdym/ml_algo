import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/from_tree_nodes_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/_helper/tree_nodes_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/_helper/get_split_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/from_predicate_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';
import 'package:ml_linalg/vector.dart';

part 'tree_node.g.dart';

@JsonSerializable()
class TreeNode {
  TreeNode(
    this.predicateType,
    this.splitValue,
    this.splitIndex,
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
    toJson: predicateTypeToJson,
    fromJson: fromPredicateTypeJson,
  )
  final PredicateType? predicateType;

  @JsonKey(name: splitValueJsonKey)
  final num? splitValue;

  @JsonKey(name: splitIndexJsonKey)
  final int? splitIndex;

  @JsonKey(name: levelJsonKey)
  final int level;

  bool get isLeaf => children == null || children!.isEmpty;

  bool get isRoot =>
      predicateType == null || splitIndex == null || splitValue == null;

  bool get isFake =>
      predicateType == null &&
      splitIndex == null &&
      splitValue == null &&
      label == null &&
      children == null;

  bool testSample(Vector sample) {
    if (isRoot) {
      return true;
    }

    final predicate = getSplitPredicateByType(predicateType!);

    return predicate(
      sample,
      splitIndex!,
      splitValue!,
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
