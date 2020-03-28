import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_serialize.dart' as leaf_label;
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_serializable_field.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_serialize.dart' as predicate_type;

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
