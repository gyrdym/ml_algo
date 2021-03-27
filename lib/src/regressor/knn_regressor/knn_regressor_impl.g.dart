// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnRegressorImpl _$KnnRegressorImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KnnRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const ['D', 'T', 'S', 'K', r'$V']);
    final val = KnnRegressorImpl(
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(
          json,
          'S',
          (v) => const KnnSolverJsonConverter()
              .fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'K', (v) => const KernelJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'D', (v) => _$enumDecode(_$DTypeEnumMap, v)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'targetName': 'T',
    'solver': 'S',
    'kernel': 'K',
    'dtype': 'D',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$KnnRegressorImplToJson(KnnRegressorImpl instance) =>
    <String, dynamic>{
      'D': _$DTypeEnumMap[instance.dtype],
      'T': instance.targetName,
      'S': const KnnSolverJsonConverter().toJson(instance.solver),
      'K': const KernelJsonConverter().toJson(instance.kernel),
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
