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
          json,
          'F',
          (v) =>
              const MatrixJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json,
          'O',
          (v) =>
              const MatrixJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'K', (v) => v as int),
      $checkedConvert(json, 'D',
          (v) => const DistanceTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'S', (v) => v as bool),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
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

Map<String, dynamic> _$KnnSolverImplToJson(KnnSolverImpl instance) =>
    <String, dynamic>{
      'F': const MatrixJsonConverter().toJson(instance.trainFeatures),
      'O': const MatrixJsonConverter().toJson(instance.trainOutcomes),
      'K': instance.k,
      'D': const DistanceTypeJsonConverter().toJson(instance.distanceType),
      'S': instance.standardize,
      r'$V': instance.schemaVersion,
    };
