import 'dart:io';

import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/helpers/validate_probability_value.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_json_keys.dart';

part 'leaf_label.g.dart';

@JsonSerializable(includeIfNull: false)
class TreeLeafLabel implements Serializable {
  TreeLeafLabel(this.value, {
    required this.probability,
    // Define schema version here because of json serializer
  }) : schemaVersion = 1 {
    validateProbabilityValue(probability);
  }

  factory TreeLeafLabel.fromJson(Map<String, dynamic> json) =>
      _$TreeLeafLabelFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$TreeLeafLabelToJson(this);

  @JsonKey(name: leafLabelValueJsonKey)
  final num value;

  @JsonKey(name: leafLabelProbabilityJsonKey)
  final num probability;

  @override
  @JsonKey(name: leafLabelSchemaVersionJsonKey)
  final int schemaVersion;

  @override
  Future<File> saveAsJson(String filePath) {
    throw UnimplementedError();
  }
}
