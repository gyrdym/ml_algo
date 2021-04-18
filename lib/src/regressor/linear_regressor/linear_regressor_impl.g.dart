// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'linear_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LinearRegressorImpl _$LinearRegressorImplFromJson(Map<String, dynamic> json) {
  return $checkedNew('LinearRegressorImpl', json, () {
    $checkKeys(json, allowedKeys: const [
      'OT',
      'IL',
      'LRT',
      'ICT',
      'ILT',
      'MCU',
      'L',
      'RT',
      'RS',
      'BS',
      'IC',
      'FDN',
      'TN',
      'FI',
      'IS',
      'CS',
      'CPI',
      'DT',
      r'$V'
    ]);
    final val = LinearRegressorImpl(
      $checkedConvert(
          json, 'CS', (v) => Vector.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'TN', (v) => v as String),
      optimizerType: $checkedConvert(
          json,
          'OT',
          (v) =>
              const LinearOptimizerTypeJsonConverter().fromJson(v as String)),
      iterationsLimit: $checkedConvert(json, 'IL', (v) => v as int),
      learningRateType: $checkedConvert(json, 'LRT',
          (v) => const LearningRateTypeJsonConverter().fromJson(v as String)),
      initialCoefficientsType: $checkedConvert(
          json,
          'ICT',
          (v) => const InitialCoefficientsTypeJsonConverter()
              .fromJson(v as String)),
      initialLearningRate: $checkedConvert(json, 'ILT', (v) => v as num),
      minCoefficientsUpdate: $checkedConvert(json, 'MCU', (v) => v as num),
      lambda: $checkedConvert(json, 'L', (v) => v as num),
      batchSize: $checkedConvert(json, 'BS', (v) => v as int),
      isFittingDataNormalized: $checkedConvert(json, 'FDN', (v) => v as bool),
      fitIntercept: $checkedConvert(json, 'FI', (v) => v as bool),
      interceptScale: $checkedConvert(json, 'IS', (v) => (v as num).toDouble()),
      dtype: $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
      regularizationType: $checkedConvert(json, 'RT',
          (v) => _$enumDecodeNullable(_$RegularizationTypeEnumMap, v)),
      randomSeed: $checkedConvert(json, 'RS', (v) => v as int?),
      initialCoefficients: $checkedConvert(
          json,
          'IC',
          (v) => const MatrixJsonConverterNullable()
              .fromJson(v as Map<String, dynamic>)),
      costPerIteration: $checkedConvert(json, 'CPI',
          (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
    );
    return val;
  }, fieldKeyMap: const {
    'coefficients': 'CS',
    'targetName': 'TN',
    'optimizerType': 'OT',
    'iterationsLimit': 'IL',
    'learningRateType': 'LRT',
    'initialCoefficientsType': 'ICT',
    'initialLearningRate': 'ILT',
    'minCoefficientsUpdate': 'MCU',
    'lambda': 'L',
    'batchSize': 'BS',
    'isFittingDataNormalized': 'FDN',
    'fitIntercept': 'FI',
    'interceptScale': 'IS',
    'dtype': 'DT',
    'schemaVersion': r'$V',
    'regularizationType': 'RT',
    'randomSeed': 'RS',
    'initialCoefficients': 'IC',
    'costPerIteration': 'CPI'
  });
}

Map<String, dynamic> _$LinearRegressorImplToJson(LinearRegressorImpl instance) {
  final val = <String, dynamic>{
    'OT':
        const LinearOptimizerTypeJsonConverter().toJson(instance.optimizerType),
    'IL': instance.iterationsLimit,
    'LRT':
        const LearningRateTypeJsonConverter().toJson(instance.learningRateType),
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull(
      'ICT',
      const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType));
  val['ILT'] = instance.initialLearningRate;
  val['MCU'] = instance.minCoefficientsUpdate;
  val['L'] = instance.lambda;
  writeNotNull('RT', _$RegularizationTypeEnumMap[instance.regularizationType]);
  writeNotNull('RS', instance.randomSeed);
  val['BS'] = instance.batchSize;
  writeNotNull('IC',
      const MatrixJsonConverterNullable().toJson(instance.initialCoefficients));
  val['FDN'] = instance.isFittingDataNormalized;
  val['TN'] = instance.targetName;
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['CS'] = instance.coefficients;
  writeNotNull('CPI', instance.costPerIteration);
  val['DT'] = const DTypeJsonConverter().toJson(instance.dtype);
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

K? _$enumDecodeNullable<K, V>(
  Map<K, V> enumValues,
  dynamic source, {
  K? unknownValue,
}) {
  if (source == null) {
    return null;
  }
  return _$enumDecode<K, V>(enumValues, source, unknownValue: unknownValue);
}

const _$RegularizationTypeEnumMap = {
  RegularizationType.L1: 'L1',
  RegularizationType.L2: 'L2',
};
