// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'softmax_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SoftmaxRegressorImpl _$SoftmaxRegressorImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('SoftmaxRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'CN',
      'FI',
      'IS',
      'CBC',
      'DT',
      'LF',
      'PL',
      'NL',
      'CPI'
    ]);
    final val = SoftmaxRegressorImpl(
      $checkedConvert(
          json, 'CBC', (v) => fromMatrixJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'CN', (v) => (v as List)?.map((e) => e as String)),
      $checkedConvert(json, 'LF', (v) => fromLinkFunctionJson(v as String)),
      $checkedConvert(json, 'FI', (v) => v as bool),
      $checkedConvert(json, 'IS', (v) => v as num),
      $checkedConvert(json, 'PL', (v) => v as num),
      $checkedConvert(json, 'NL', (v) => v as num),
      $checkedConvert(
          json, 'CPI', (v) => (v as List)?.map((e) => e as num)?.toList()),
      $checkedConvert(json, 'DT', (v) => fromDTypeJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {
    'coefficientsByClasses': 'CBC',
    'targetNames': 'CN',
    'linkFunction': 'LF',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'positiveLabel': 'PL',
    'negativeLabel': 'NL',
    'costPerIteration': 'CPI',
    'dtype': 'DT'
  });
}

Map<String, dynamic> _$SoftmaxRegressorImplToJson(
    SoftmaxRegressorImpl instance) {
  final val = <String, dynamic>{
    'CN': instance.targetNames?.toList(),
    'FI': instance.fitIntercept,
    'IS': instance.interceptScale,
    'CBC': matrixToJson(instance.coefficientsByClasses),
    'DT': dTypeToJson(instance.dtype),
    'LF': linkFunctionToJson(instance.linkFunction),
    'PL': instance.positiveLabel,
    'NL': instance.negativeLabel,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('CPI', instance.costPerIteration);
  return val;
}
