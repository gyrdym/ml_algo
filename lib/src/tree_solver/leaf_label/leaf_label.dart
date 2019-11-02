import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';

class TreeLeafLabel with SerializableMixin {
  TreeLeafLabel(this.value, {this.probability});

  final double value;
  final double probability;

  @override
  Map<String, dynamic> serialize() => <String, dynamic>{
    'value': value,
    'probability': probability,
  };
}
