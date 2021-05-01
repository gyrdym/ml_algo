// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnClassifierImpl _$KnnClassifierImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KnnClassifierImpl', json, () {
    $checkKeys(json, allowedKeys: const ['T', 'D', 'C', 'K', 'S', 'P', r'$V']);
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
      $checkedConvert(
          json, 'D', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
    );
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

Map<String, dynamic> _$KnnClassifierImplToJson(KnnClassifierImpl instance) {
  final val = <String, dynamic>{
    'T': instance.targetColumnName,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('D', const DTypeJsonConverter().toJson(instance.dtype));
  val['C'] = instance.classLabels;
  writeNotNull('K', const KernelJsonConverter().toJson(instance.kernel));
  writeNotNull('S', const KnnSolverJsonConverter().toJson(instance.solver));
  val['P'] = instance.classLabelPrefix;
  writeNotNull(r'$V', instance.schemaVersion);
  return val;
}
