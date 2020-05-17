// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'logistic_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LogisticRegressorImpl _$LogisticRegressorImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('LogisticRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'CBC',
      'CN',
      'FI',
      'IS',
      'DT',
      'PT',
      'PL',
      'NL',
      'LF'
    ]);
    final val = LogisticRegressorImpl(
      $checkedConvert(json, 'CN', (v) => (v as List)?.map((dynamic e) => e as String)),
      $checkedConvert(json, 'LF', (v) => fromLinkFunctionJson(v as String)),
      $checkedConvert(json, 'FI', (v) => v as bool),
      $checkedConvert(json, 'IS', (v) => v as num),
      $checkedConvert(
          json, 'CBC', (v) => fromMatrixJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'PT', (v) => v as num),
      $checkedConvert(json, 'NL', (v) => v as num),
      $checkedConvert(json, 'PL', (v) => v as num),
      $checkedConvert(json, 'DT', (v) => fromDTypeJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {
    'classNames': 'CN',
    'linkFunction': 'LF',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'coefficientsByClasses': 'CBC',
    'probabilityThreshold': 'PT',
    'negativeLabel': 'NL',
    'positiveLabel': 'PL',
    'dtype': 'DT'
  });
}

Map<String, dynamic> _$LogisticRegressorImplToJson(
        LogisticRegressorImpl instance) =>
    <String, dynamic>{
      'CBC': matrixToJson(instance.coefficientsByClasses),
      'CN': instance.classNames?.toList(),
      'FI': instance.fitIntercept,
      'IS': instance.interceptScale,
      'DT': dTypeToJson(instance.dtype),
      'PT': instance.probabilityThreshold,
      'PL': instance.positiveLabel,
      'NL': instance.negativeLabel,
      'LF': linkFunctionToJson(instance.linkFunction),
    };
