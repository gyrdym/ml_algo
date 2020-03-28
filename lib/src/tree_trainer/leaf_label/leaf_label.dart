import 'package:ml_algo/src/helpers/validate_probability_value.dart';

class TreeLeafLabel {
  TreeLeafLabel(this.value, {this.probability}) {
    if (probability != null) {
      validateProbabilityValue(probability);
    }
  }

  final num value;
  final num probability;
}
