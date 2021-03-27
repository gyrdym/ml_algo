// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'decision_tree_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DecisionTreeClassifierImpl _$DecisionTreeClassifierImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('DecisionTreeClassifierImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'E',
      'S',
      'D',
      'DT',
      'T',
      'R',
      'positiveLabel',
      'negativeLabel',
      r'$V'
    ]);
    final val = DecisionTreeClassifierImpl(
      $checkedConvert(json, 'E', (v) => v as num),
      $checkedConvert(json, 'S', (v) => v as int),
      $checkedConvert(json, 'D', (v) => v as int),
      $checkedConvert(
          json, 'R', (v) => fromTreeNodeJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(json, 'DT', (v) => _$enumDecode(_$DTypeEnumMap, v)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    $checkedConvert(json, 'positiveLabel', (v) => val.positiveLabel = v as num);
    $checkedConvert(json, 'negativeLabel', (v) => val.negativeLabel = v as num);
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
      'DT': _$DTypeEnumMap[instance.dtype],
      'T': instance.targetColumnName,
      'R': treeNodeToJson(instance.treeRootNode),
      'positiveLabel': instance.positiveLabel,
      'negativeLabel': instance.negativeLabel,
      r'$V': instance.schemaVersion,
    };

K _$enumDecode<K, V>(
  Map<K, V> enumValues,
  Object? source, {
  K? unknownValue,
}) {
  if (source == null) {
    throw ArgumentError(
      'A value must be provided. Supported values: '
      '${enumValues.values.join(', ')}',
    );
  }

  return enumValues.entries.singleWhere(
    (e) => e.value == source,
    orElse: () {
      if (unknownValue == null) {
        throw ArgumentError(
          '`$source` is not one of the supported values: '
          '${enumValues.values.join(', ')}',
        );
      }
      return MapEntry(unknownValue, enumValues.values.first);
    },
  ).key;
}

const _$DTypeEnumMap = {
  DType.float32: 'float32',
  DType.float64: 'float64',
};
