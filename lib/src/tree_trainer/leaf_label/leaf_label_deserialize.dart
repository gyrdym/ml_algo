import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_serializable_field.dart';

TreeLeafLabel deserialize(Map<String, dynamic> serialized) {
  final value = serialized[valueField] as num;
  final probability = serialized[probabilityField] as num;

  return TreeLeafLabel(
    value,
    probability: probability,
  );
}
