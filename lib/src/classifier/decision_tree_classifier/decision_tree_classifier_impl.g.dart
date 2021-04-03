// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'decision_tree_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DecisionTreeClassifierImpl _$DecisionTreeClassifierImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('DecisionTreeClassifierImpl', json, () {
    $checkKeys(json, allowedKeys: const ['E', 'S', 'D', 'DT', 'T', 'R', r'$V']);
    final val = DecisionTreeClassifierImpl(
      $checkedConvert(json, 'E', (v) => v as num),
      $checkedConvert(json, 'S', (v) => v as int),
      $checkedConvert(json, 'D', (v) => v as int),
      $checkedConvert(
          json, 'R', (v) => fromTreeNodeJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
    );
    return val;
  }, fieldKeyMap: const {
    'minError': 'E',
    'minSamplesCount': 'S',
    'maxDepth': 'D',
    'treeRootNode': 'R',
    'targetColumnName': 'T',
    'dtype': 'DT',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$DecisionTreeClassifierImplToJson(
        DecisionTreeClassifierImpl instance) =>
    <String, dynamic>{
      'E': instance.minError,
      'S': instance.minSamplesCount,
      'D': instance.maxDepth,
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      'T': instance.targetColumnName,
      'R': treeNodeToJson(instance.treeRootNode),
      r'$V': instance.schemaVersion,
    };
