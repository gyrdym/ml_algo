// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'decision_tree_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DecisionTreeClassifierImpl _$DecisionTreeClassifierImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('DecisionTreeClassifierImpl', json, () {
    $checkKeys(json,
        allowedKeys: const ['E', 'S', 'D', 'DT', 'T', 'I', 'R', r'$V']);
    final val = DecisionTreeClassifierImpl(
      $checkedConvert(json, 'E', (v) => v as num),
      $checkedConvert(json, 'S', (v) => v as int),
      $checkedConvert(json, 'D', (v) => v as int),
      $checkedConvert(
          json, 'R', (v) => TreeNode.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(json, 'I', (v) => v as int),
      $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'minError': 'E',
    'minSamplesCount': 'S',
    'maxDepth': 'D',
    'treeRootNode': 'R',
    'targetColumnName': 'T',
    'targetColumnIndex': 'I',
    'dtype': 'DT',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$DecisionTreeClassifierImplToJson(
    DecisionTreeClassifierImpl instance) {
  final val = <String, dynamic>{
    'E': instance.minError,
    'S': instance.minSamplesCount,
    'D': instance.maxDepth,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('DT', const DTypeJsonConverter().toJson(instance.dtype));
  val['T'] = instance.targetColumnName;
  val['I'] = instance.targetColumnIndex;
  val['R'] = instance.treeRootNode.toJson();
  val[r'$V'] = instance.schemaVersion;
  return val;
}
