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
          json, 'DT', (v) => _$enumDecodeNullable(_$DTypeEnumMap, v)),
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
      'DT': _$DTypeEnumMap[instance.dtype],
      'T': instance.targetColumnName,
      'R': treeNodeToJson(instance.treeRootNode),
    };

T _$enumDecode<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    throw ArgumentError('A value must be provided. Supported values: '
        '${enumValues.values.join(', ')}');
  }

  final value = enumValues.entries
      .singleWhere((e) => e.value == source, orElse: () => null)
      ?.key;

  if (value == null && unknownValue == null) {
    throw ArgumentError('`$source` is not one of the supported values: '
        '${enumValues.values.join(', ')}');
  }
  return value ?? unknownValue;
}

T _$enumDecodeNullable<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    return null;
  }
  return _$enumDecode<T>(enumValues, source, unknownValue: unknownValue);
}

const _$DTypeEnumMap = {
  DType.float32: 'float32',
  DType.float64: 'float64',
};
