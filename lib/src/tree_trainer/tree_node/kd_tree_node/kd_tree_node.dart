import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/_helpers/from_kd_tree_nodes_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/_helpers/kd_tree_nodes_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node_json_keys.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/_helper/get_tree_node_splitting_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/from_tree_node_splitting_predicate_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_to_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'kd_tree_node.g.dart';

@JsonSerializable()
class KDTreeNode implements TreeNode {
  KDTreeNode(
    this.predicateType,
    this.splittingValue,
    this.splittingIndex,
    this.children,
    this.samples, [
    this.level = 0,
  ]);

  factory KDTreeNode.fromJson(Map<String, dynamic> json) =>
      _$KDTreeNodeFromJson(json);

  Map<String, dynamic> toJson() => _$KDTreeNodeToJson(this);

  @override
  @JsonKey(
    name: kdTreeNodeChildrenJsonKey,
    toJson: kdTreeNodesToJson,
    fromJson: fromKDTreeNodesJson,
  )
  final List<KDTreeNode>? children;

  @JsonKey(name: kdTreeNodeSamplesJsonKey)
  final Matrix? samples;

  @override
  @JsonKey(
    name: kdTreeNodePredicateTypeJsonKey,
    toJson: splittingPredicateTypeToJson,
    fromJson: fromSplittingPredicateTypeJson,
  )
  final TreeNodeSplittingPredicateType? predicateType;

  @override
  @JsonKey(name: kdTreeNodeSplittingValueJsonKey)
  final num? splittingValue;

  @override
  @JsonKey(name: kdTreeNodeSplittingIndexJsonKey)
  final int? splittingIndex;

  @override
  @JsonKey(name: kdTreeNodeLevelJsonKey)
  final int level;

  bool get isLeaf => children == null || children!.isEmpty;

  bool get isRoot => splittingValue == null;

  bool testSample(Vector sample) {
    if (isRoot) {
      return true;
    }

    return getTreeNodeSplittingPredicateByType(predicateType!)(
      sample,
      splittingIndex!,
      splittingValue!,
    );
  }
}
