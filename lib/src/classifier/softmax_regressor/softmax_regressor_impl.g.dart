// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'softmax_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SoftmaxRegressorImpl _$SoftmaxRegressorImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('SoftmaxRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'OT',
      'IL',
      'ILR',
      'MCU',
      'L',
      'RT',
      'RS',
      'BS',
      'FDN',
      'LR',
      'ICT',
      'IC',
      'CN',
      'FI',
      'IS',
      'CBC',
      'DT',
      'LF',
      'PL',
      'NL',
      'CPI',
      r'$V'
    ]);
    final val = SoftmaxRegressorImpl(
      $checkedConvert(
          json, 'OT', (v) => _$enumDecode(_$LinearOptimizerTypeEnumMap, v)),
      $checkedConvert(json, 'IL', (v) => v as int),
      $checkedConvert(json, 'ILR', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'MCU', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'L', (v) => (v as num).toDouble()),
      $checkedConvert(
          json,
          'RT',
          (v) =>
              const RegularizationTypeJsonConverter().fromJson(v as String?)),
      $checkedConvert(json, 'RS', (v) => v as int?),
      $checkedConvert(json, 'BS', (v) => v as int),
      $checkedConvert(json, 'FDN', (v) => v as bool),
      $checkedConvert(
          json, 'LR', (v) => _$enumDecode(_$LearningRateTypeEnumMap, v)),
      $checkedConvert(json, 'ICT',
          (v) => _$enumDecode(_$InitialCoefficientsTypeEnumMap, v)),
      $checkedConvert(
          json,
          'IC',
          (v) =>
              const MatrixJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'CBC', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'CN', (v) => (v as List<dynamic>).map((e) => e as String)),
      $checkedConvert(json, 'LF', (v) => fromLinkFunctionJson(v as String)),
      $checkedConvert(json, 'FI', (v) => v as bool),
      $checkedConvert(json, 'IS', (v) => v as num),
      $checkedConvert(json, 'PL', (v) => v as num),
      $checkedConvert(json, 'NL', (v) => v as num),
      $checkedConvert(json, 'CPI',
          (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
      $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'optimizerType': 'OT',
    'iterationsLimit': 'IL',
    'initialLearningRate': 'ILR',
    'minCoefficientsUpdate': 'MCU',
    'lambda': 'L',
    'regularizationType': 'RT',
    'randomSeed': 'RS',
    'batchSize': 'BS',
    'isFittingDataNormalized': 'FDN',
    'learningRateType': 'LR',
    'initialCoefficientsType': 'ICT',
    'initialCoefficients': 'IC',
    'coefficientsByClasses': 'CBC',
    'targetNames': 'CN',
    'linkFunction': 'LF',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'positiveLabel': 'PL',
    'negativeLabel': 'NL',
    'costPerIteration': 'CPI',
    'dtype': 'DT',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$SoftmaxRegressorImplToJson(
    SoftmaxRegressorImpl instance) {
  final val = <String, dynamic>{
    'OT': _$LinearOptimizerTypeEnumMap[instance.optimizerType],
    'IL': instance.iterationsLimit,
    'ILR': instance.initialLearningRate,
    'MCU': instance.minCoefficientsUpdate,
    'L': instance.lambda,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull(
      'RT',
      const RegularizationTypeJsonConverter()
          .toJson(instance.regularizationType));
  writeNotNull('RS', instance.randomSeed);
  val['BS'] = instance.batchSize;
  val['FDN'] = instance.isFittingDataNormalized;
  val['LR'] = _$LearningRateTypeEnumMap[instance.learningRateType];
  val['ICT'] =
      _$InitialCoefficientsTypeEnumMap[instance.initialCoefficientsType];
  writeNotNull(
      'IC', const MatrixJsonConverter().toJson(instance.initialCoefficients));
  val['CN'] = instance.targetNames.toList();
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['CBC'] = instance.coefficientsByClasses;
  val['DT'] = const DTypeJsonConverter().toJson(instance.dtype);
  val['LF'] = linkFunctionToJson(instance.linkFunction);
  val['PL'] = instance.positiveLabel;
  val['NL'] = instance.negativeLabel;
  writeNotNull('CPI', instance.costPerIteration);
  val[r'$V'] = instance.schemaVersion;
  return val;
}

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

const _$LinearOptimizerTypeEnumMap = {
  LinearOptimizerType.gradient: 'gradient',
  LinearOptimizerType.coordinate: 'coordinate',
};

const _$LearningRateTypeEnumMap = {
  LearningRateType.decreasingAdaptive: 'decreasingAdaptive',
  LearningRateType.constant: 'constant',
};

const _$InitialCoefficientsTypeEnumMap = {
  InitialCoefficientsType.zeroes: 'zeroes',
};
