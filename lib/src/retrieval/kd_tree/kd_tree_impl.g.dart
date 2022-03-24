// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeImpl _$KDTreeImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KDTreeImpl', json, () {
    $checkKeys(json, allowedKeys: const ['L', 'R', 'D']);
    final val = KDTreeImpl(
      $checkedConvert(json, 'L', (v) => v as int),
      $checkedConvert(
          json, 'R', (v) => KDTreeNode.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'D', (v) => _$enumDecode(_$DTypeEnumMap, v)),
    );
    return val;
  }, fieldKeyMap: const {'leafSize': 'L', 'root': 'R', 'dtype': 'D'});
}

Map<String, dynamic> _$KDTreeImplToJson(KDTreeImpl instance) =>
    <String, dynamic>{
      'L': instance.leafSize,
      'R': instance.root.toJson(),
      'D': _$DTypeEnumMap[instance.dtype],
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
