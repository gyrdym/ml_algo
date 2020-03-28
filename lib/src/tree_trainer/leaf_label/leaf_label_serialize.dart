import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_serializable_field.dart';

Map<String, dynamic> serialize(TreeLeafLabel label) => <String, dynamic>{
  valueField: label.value,
  probabilityField: label.probability,
};
