// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'logistic_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LogisticRegressorImpl _$LogisticRegressorImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('LogisticRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'O',
      'I',
      'LR',
      'U',
      'L',
      'R',
      'RS',
      'B',
      'N',
      'LRT',
      'ICT',
      'IC',
      'CBC',
      'CN',
      'FI',
      'IS',
      'DT',
      'PT',
      'PL',
      'NL',
      'LF',
      'CPI',
      r'$V'
    ]);
    final val = LogisticRegressorImpl(
      $checkedConvert(
          json, 'O', (v) => _$enumDecode(_$LinearOptimizerTypeEnumMap, v)),
      $checkedConvert(json, 'I', (v) => v as int),
      $checkedConvert(json, 'LR', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'U', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'L', (v) => (v as num).toDouble()),
      $checkedConvert(
          json,
          'R',
          (v) =>
              const RegularizationTypeJsonConverter().fromJson(v as String?)),
      $checkedConvert(json, 'RS', (v) => v as int?),
      $checkedConvert(json, 'B', (v) => v as int),
      $checkedConvert(json, 'N', (v) => v as bool),
      $checkedConvert(
          json, 'LRT', (v) => _$enumDecode(_$LearningRateTypeEnumMap, v)),
      $checkedConvert(json, 'ICT',
          (v) => _$enumDecode(_$InitialCoefficientsTypeEnumMap, v)),
      $checkedConvert(
          json,
          'IC',
          (v) =>
              const VectorJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'CN', (v) => (v as List<dynamic>).map((e) => e as String)),
      $checkedConvert(json, 'LF', (v) => fromLinkFunctionJson(v as String)),
      $checkedConvert(json, 'FI', (v) => v as bool),
      $checkedConvert(json, 'IS', (v) => v as num),
      $checkedConvert(
          json, 'CBC', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'PT', (v) => v as num),
      $checkedConvert(json, 'NL', (v) => v as num),
      $checkedConvert(json, 'PL', (v) => v as num),
      $checkedConvert(json, 'CPI',
          (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
      $checkedConvert(json, 'DT', (v) => _$enumDecode(_$DTypeEnumMap, v)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'optimizerType': 'O',
    'iterationsLimit': 'I',
    'initialLearningRate': 'LR',
    'minCoefficientsUpdate': 'U',
    'lambda': 'L',
    'regularizationType': 'R',
    'randomSeed': 'RS',
    'batchSize': 'B',
    'isFittingDataNormalized': 'N',
    'learningRateType': 'LRT',
    'initialCoefficientsType': 'ICT',
    'initialCoefficients': 'IC',
    'targetNames': 'CN',
    'linkFunction': 'LF',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'coefficientsByClasses': 'CBC',
    'probabilityThreshold': 'PT',
    'negativeLabel': 'NL',
    'positiveLabel': 'PL',
    'costPerIteration': 'CPI',
    'dtype': 'DT',
    'schemaVersion': r'$V'
  });
}

Map<String, dynamic> _$LogisticRegressorImplToJson(
    LogisticRegressorImpl instance) {
  final val = <String, dynamic>{
    'O': _$LinearOptimizerTypeEnumMap[instance.optimizerType],
    'I': instance.iterationsLimit,
    'LR': instance.initialLearningRate,
    'U': instance.minCoefficientsUpdate,
    'L': instance.lambda,
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull(
      'R',
      const RegularizationTypeJsonConverter()
          .toJson(instance.regularizationType));
  writeNotNull('RS', instance.randomSeed);
  val['B'] = instance.batchSize;
  val['N'] = instance.isFittingDataNormalized;
  val['LRT'] = _$LearningRateTypeEnumMap[instance.learningRateType];
  val['ICT'] =
      _$InitialCoefficientsTypeEnumMap[instance.initialCoefficientsType];
  writeNotNull(
      'IC', const VectorJsonConverter().toJson(instance.initialCoefficients));
  val['CBC'] = instance.coefficientsByClasses;
  val['CN'] = instance.targetNames.toList();
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['DT'] = _$DTypeEnumMap[instance.dtype];
  val['PT'] = instance.probabilityThreshold;
  val['PL'] = instance.positiveLabel;
  val['NL'] = instance.negativeLabel;
  val['LF'] = linkFunctionToJson(instance.linkFunction);
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

const _$DTypeEnumMap = {
  DType.float32: 'float32',
  DType.float64: 'float64',
};
