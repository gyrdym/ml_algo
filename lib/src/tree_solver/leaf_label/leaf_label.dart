import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_probability_value.dart';

class TreeLeafLabel with SerializableMixin {
  TreeLeafLabel(this.value, {this.probability}) {
    if (probability != null) {
      validateProbabilityValue(probability);
    }
  }

  final num value;
  final num probability;

  @override
  Map<String, dynamic> serialize() => <String, dynamic>{
    'value': value,
    'probability': probability,
  };
}
