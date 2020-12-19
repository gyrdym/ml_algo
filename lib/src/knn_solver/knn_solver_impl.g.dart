// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_solver_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnSolverImpl _$KnnSolverImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('KnnSolverImpl', json, () {
    $checkKeys(json, allowedKeys: const ['F', 'O', 'K', 'D', 'S']);
    final val = KnnSolverImpl(
      $checkedConvert(
          json, 'F', (v) => fromMatrixJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'O', (v) => fromMatrixJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'K', (v) => v as int),
      $checkedConvert(json, 'D', (v) => fromDistanceTypeJson(v as String)),
      $checkedConvert(json, 'S', (v) => v as bool),
    );
    return val;
  }, fieldKeyMap: const {
    'trainFeatures': 'F',
    'trainOutcomes': 'O',
    'k': 'K',
    'distanceType': 'D',
    'standardize': 'S'
  });
}

Map<String, dynamic> _$KnnSolverImplToJson(KnnSolverImpl instance) =>
    <String, dynamic>{
      'F': matrixToJson(instance.trainFeatures),
      'O': matrixToJson(instance.trainOutcomes),
      'K': instance.k,
      'D': distanceTypeToJson(instance.distanceType),
      'S': instance.standardize,
    };
