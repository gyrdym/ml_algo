// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'linear_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LinearRegressorImpl _$LinearRegressorImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('LinearRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const ['TN', 'FI', 'IS', 'CS', 'DT']);
    final val = LinearRegressorImpl(
      $checkedConvert(
          json, 'CS', (v) => fromVectorJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'TN', (v) => v as String),
      fitIntercept: $checkedConvert(json, 'FI', (v) => v as bool),
      interceptScale:
          $checkedConvert(json, 'IS', (v) => (v as num)?.toDouble()),
      dtype: $checkedConvert(json, 'DT', (v) => fromDTypeJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {
    'coefficients': 'CS',
    'targetName': 'TN',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'dtype': 'DT'
  });
}

Map<String, dynamic> _$LinearRegressorImplToJson(
        LinearRegressorImpl instance) =>
    <String, dynamic>{
      'TN': instance.targetName,
      'FI': instance.fitIntercept,
      'IS': instance.interceptScale,
      'CS': vectorToJson(instance.coefficients),
      'DT': dTypeToJson(instance.dtype),
    };
