import 'package:ml_algo/src/common/serializable/serializer.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_serializable_field.dart';

class TreeLeafLabelSerializer implements Serializer<TreeLeafLabel> {

  const TreeLeafLabelSerializer();

  @override
  Map<String, dynamic> serialize(TreeLeafLabel label) => <String, dynamic>{
    valueField: label.value,
    probabilityField: label.probability,
  };

  @override
  TreeLeafLabel deserialize(Map<String, dynamic> serialized) {
    final value = serialized[valueField] as num;
    final probability = serialized[probabilityField] as num;

    return TreeLeafLabel(
      value,
      probability: probability,
    );
  }
}
