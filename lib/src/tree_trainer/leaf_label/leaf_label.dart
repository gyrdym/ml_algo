import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/helpers/validate_probability_value.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_json_keys.dart';

part 'leaf_label.g.dart';

@JsonSerializable(includeIfNull: false)
class TreeLeafLabel {
  TreeLeafLabel(this.value, {this.probability}) {
    if (probability != null) {
      validateProbabilityValue(probability);
    }
  }

  factory TreeLeafLabel.fromJson(Map<String, dynamic> json) =>
      _$TreeLeafLabelFromJson(json);
  Map<String, dynamic> toJson() => _$TreeLeafLabelToJson(this);

  @JsonKey(name: valueJsonKey)
  final num value;

  @JsonKey(name: probabilityJsonKey)
  final num probability;
}
