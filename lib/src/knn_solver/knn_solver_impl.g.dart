// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_solver_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnSolverImpl _$KnnSolverImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KnnSolverImpl', json, () {
    $checkKeys(json, allowedKeys: const ['F', 'O', 'K', 'D', 'S', r'$V']);
    final val = KnnSolverImpl(
      $checkedConvert(
          json, 'F', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'O', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'K', (v) => v as int),
      $checkedConvert(json, 'D',
          (v) => const DistanceTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'S', (v) => v as bool),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
    );
    return val;
  }, fieldKeyMap: const {
    'trainFeatures': 'F',
    'trainOutcomes': 'O',
    'k': 'K',
    'distanceType': 'D',
    'standardize': 'S',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$KnnSolverImplToJson(KnnSolverImpl instance) {
  final val = <String, dynamic>{
    'F': instance.trainFeatures.toJson(),
    'O': instance.trainOutcomes.toJson(),
    'K': instance.k,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull(
      'D', const DistanceTypeJsonConverter().toJson(instance.distanceType));
  val['S'] = instance.standardize;
  writeNotNull(r'$V', instance.schemaVersion);
  return val;
}
