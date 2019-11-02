import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';

class TreeLeafLabel with SerializableMixin {
  TreeLeafLabel(this.value, {this.probability}) {
    if (probability != null && (probability < 0 || probability > 1)) {
      throw Exception('Probability value should be within the range 0..1 '
          '(both inclusive)');
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
