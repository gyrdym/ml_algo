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
      $checkedConvert(
          json, 'D', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
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

Map<String, dynamic> _$KnnRegressorImplToJson(KnnRegressorImpl instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('D', const DTypeJsonConverter().toJson(instance.dtype));
  val['T'] = instance.targetName;
  writeNotNull('S', const KnnSolverJsonConverter().toJson(instance.solver));
  writeNotNull('K', const KernelJsonConverter().toJson(instance.kernel));
  writeNotNull(r'$V', instance.schemaVersion);
  return val;
}
