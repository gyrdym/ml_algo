// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'leaf_label.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TreeLeafLabel _$TreeLeafLabelFromJson(Map<String, dynamic> json) {
  return $checkedNew('TreeLeafLabel', json, () {
    $checkKeys(json, allowedKeys: const ['V', 'P']);
    final val = TreeLeafLabel(
      $checkedConvert(json, 'V', (v) => v as num),
      probability: $checkedConvert(json, 'P', (v) => v as num),
    );
    return val;
  }, fieldKeyMap: const {'value': 'V', 'probability': 'P'});
}

Map<String, dynamic> _$TreeLeafLabelToJson(TreeLeafLabel instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('V', instance.value);
  writeNotNull('P', instance.probability);
  return val;
}
