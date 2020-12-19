// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'decision_tree_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DecisionTreeClassifierImpl _$DecisionTreeClassifierImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('DecisionTreeClassifierImpl', json, () {
    $checkKeys(json, allowedKeys: const ['DT', 'T', 'R']);
    final val = DecisionTreeClassifierImpl(
      $checkedConvert(
          json, 'R', (v) => fromTreeNodeJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {
    'treeRootNode': 'R',
    'targetColumnName': 'T',
    'dtype': 'DT'
  });
}

Map<String, dynamic> _$DecisionTreeClassifierImplToJson(
        DecisionTreeClassifierImpl instance) =>
    <String, dynamic>{
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      'T': instance.targetColumnName,
      'R': treeNodeToJson(instance.treeRootNode),
    };
