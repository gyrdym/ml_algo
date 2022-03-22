import 'package:json_annotation/json_annotation.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

import 'kd_tree_node_json_keys.dart';

part 'kd_tree_node.g.dart';

@JsonSerializable()
class KDTreeNode {
  KDTreeNode({this.value, this.index, this.left, this.right, this.samples});

  factory KDTreeNode.fromJson(Map<String, dynamic> json) =>
      _$KDTreeNodeFromJson(json);

  Map<String, dynamic> toJson() => _$KDTreeNodeToJson(this);

  @JsonKey(name: kdTreeNodeValueJsonKey)
  final Vector? value;

  @JsonKey(name: kdTreeNodeIndexJsonKey)
  final int? index;

  @JsonKey(name: kdTreeNodeLeftJsonKey)
  final KDTreeNode? left;

  @JsonKey(name: kdTreeNodeRightJsonKey)
  final KDTreeNode? right;

  @JsonKey(name: kdTreeNodeSamplesJsonKey)
  final Matrix? samples;

  bool get isLeaf => samples != null;
}
