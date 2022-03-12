import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_point.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_tree_constants.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_tree_json_keys.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

part 'kd_tree_impl.g.dart';

@JsonSerializable()
class KDTreeImpl with SerializableMixin implements KDTree {
  KDTreeImpl(this.root, this.leafSize, this.dtype)
      : schemaVersion = kdTreeJsonSchemaVersion;

  factory KDTreeImpl.fromJson(Map<String, dynamic> json) =>
      _$KDTreeImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KDTreeImplToJson(this);

  @override
  @JsonKey(name: kdTreeLeafSizeJsonKey)
  final int leafSize;

  @override
  @JsonKey(name: kdTreeDTypeJsonKey)
  final DType dtype;

  @JsonKey(name: kdTreeRootJsonKey)
  final KDTreeNode root;

  @override
  final int schemaVersion;

  @override
  Iterable<KDPoint> query(Vector sample, int k) {
    throw UnimplementedError();
  }
}
