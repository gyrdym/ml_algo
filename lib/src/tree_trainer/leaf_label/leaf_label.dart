import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/helpers/validate_probability_value.dart';

part 'leaf_label.g.dart';

@JsonSerializable()
class TreeLeafLabel {
  TreeLeafLabel(this.value, {this.probability}) {
    if (probability != null) {
      validateProbabilityValue(probability);
    }
  }

  factory TreeLeafLabel.fromJson(Map<String, dynamic> json) =>
      _$TreeLeafLabelFromJson(json);
  Map<String, dynamic> toJson() => _$TreeLeafLabelToJson(this);

  @JsonKey(name: 'V')
  final num value;

  @JsonKey(name: 'P')
  final num probability;
}
