import 'package:json_annotation/json_annotation.dart';

import 'kd_tree_node_json_keys.dart';

part 'kd_tree_node.g.dart';

@JsonSerializable()
class KDTreeNode {
  KDTreeNode(
      {this.splitIndex, this.left, this.right, required this.pointIndices});

  factory KDTreeNode.fromJson(Map<String, dynamic> json) =>
      _$KDTreeNodeFromJson(json);

  Map<String, dynamic> toJson() => _$KDTreeNodeToJson(this);

  @JsonKey(name: kdTreeNodeIndexJsonKey)
  final int? splitIndex;

  @JsonKey(name: kdTreeNodeLeftJsonKey)
  final KDTreeNode? left;

  @JsonKey(name: kdTreeNodeRightJsonKey)
  final KDTreeNode? right;

  @JsonKey(name: kdTreeNodePointsJsonKey)
  final List<int> pointIndices;

  bool get isLeaf => left == null && right == null;
}
