// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnClassifierImpl _$KnnClassifierImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KnnClassifierImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'T',
      'D',
      'C',
      'K',
      'S',
      'P',
      'positiveLabel',
      'negativeLabel',
      r'$V'
    ]);
    final val = KnnClassifierImpl(
      $checkedConvert(json, 'T', (v) => v as String),
      $checkedConvert(
          json, 'C', (v) => (v as List<dynamic>).map((e) => e as num).toList()),
      $checkedConvert(
          json, 'K', (v) => const KernelJsonConverter().fromJson(v as String)),
      $checkedConvert(
          json,
          'S',
          (v) => const KnnSolverJsonConverter()
              .fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'P', (v) => v as String),
      $checkedConvert(json, 'D', (v) => _$enumDecode(_$DTypeEnumMap, v)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    $checkedConvert(json, 'positiveLabel', (v) => val.positiveLabel = v as num);
    $checkedConvert(json, 'negativeLabel', (v) => val.negativeLabel = v as num);
    return val;
  }, fieldKeyMap: const {
    'targetColumnName': 'T',
    'classLabels': 'C',
    'kernel': 'K',
    'solver': 'S',
    'classLabelPrefix': 'P',
    'dtype': 'D',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$KnnClassifierImplToJson(KnnClassifierImpl instance) =>
    <String, dynamic>{
      'T': instance.targetColumnName,
      'D': _$DTypeEnumMap[instance.dtype],
      'C': instance.classLabels,
      'K': const KernelJsonConverter().toJson(instance.kernel),
      'S': const KnnSolverJsonConverter().toJson(instance.solver),
      'P': instance.classLabelPrefix,
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
