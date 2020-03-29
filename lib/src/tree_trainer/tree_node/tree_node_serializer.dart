import 'package:ml_algo/src/common/serializable/serializer.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_deserialize.dart' as leaf_label;
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_serialize.dart' as leaf_label;
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_deserialize.dart' as predicate_type;
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_serialize.dart' as predicate_type;
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_serializable_field.dart';

class TreeNodeSerializer implements Serializer<TreeNode> {

  const TreeNodeSerializer();

  @override
  Map<String, dynamic> serialize(TreeNode node) => <String, dynamic>{
    predicateTypeField: predicate_type.serialize(node.predicateType),
    splittingValueField: node.splittingValue,
    splittingIndexField: node.splittingIndex,
    levelField: node.level,
    labelField: node.label != null
        ? leaf_label.serialize(node.label)
        : null,
    childrenField: node.children?.map(serialize)?.toList(),
  };

  @override
  TreeNode deserialize(Map<String, dynamic> serialized) {
    final predicateType = predicate_type
        .deserialize(serialized[predicateTypeField] as String);
    final splittingValue = serialized[splittingValueField] as num;
    final splittingIndex = serialized[splittingIndexField] as int;
    final level = serialized[levelField] as int;
    final label = leaf_label
        .deserialize(serialized[labelField] as Map<String, dynamic>);

    final serializedChildren = (serialized[childrenField] ??
        <Map<String, dynamic>>[]) as List<Map<String, dynamic>>;

    final children = serializedChildren
        .map(deserialize)
        .toList(growable: false);

    return TreeNode(
      predicateType,
      splittingValue,
      splittingIndex,
      children,
      label,
      level,
    );
  }
}
