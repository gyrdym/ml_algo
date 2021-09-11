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
      'D',
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
      decay: $checkedConvert(json, 'D', (v) => v as num),
      minCoefficientsUpdate: $checkedConvert(json, 'MCU', (v) => v as num),
      lambda: $checkedConvert(json, 'L', (v) => v as num),
      batchSize: $checkedConvert(json, 'BS', (v) => v as int),
      isFittingDataNormalized: $checkedConvert(json, 'FDN', (v) => v as bool),
      fitIntercept: $checkedConvert(json, 'FI', (v) => v as bool),
      interceptScale: $checkedConvert(json, 'IS', (v) => (v as num).toDouble()),
      dtype: $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
      regularizationType: $checkedConvert(
          json,
          'RT',
          (v) => const RegularizationTypeJsonConverterNullable()
              .fromJson(v as String?)),
      randomSeed: $checkedConvert(json, 'RS', (v) => v as int?),
      initialCoefficients: $checkedConvert(json, 'IC',
          (v) => v == null ? null : Matrix.fromJson(v as Map<String, dynamic>)),
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
    'decay': 'D',
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
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('OT',
      const LinearOptimizerTypeJsonConverter().toJson(instance.optimizerType));
  val['IL'] = instance.iterationsLimit;
  writeNotNull('LRT',
      const LearningRateTypeJsonConverter().toJson(instance.learningRateType));
  writeNotNull(
      'ICT',
      const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType));
  val['ILT'] = instance.initialLearningRate;
  val['D'] = instance.decay;
  val['MCU'] = instance.minCoefficientsUpdate;
  val['L'] = instance.lambda;
  writeNotNull(
      'RT',
      const RegularizationTypeJsonConverterNullable()
          .toJson(instance.regularizationType));
  writeNotNull('RS', instance.randomSeed);
  val['BS'] = instance.batchSize;
  writeNotNull('IC', instance.initialCoefficients?.toJson());
  val['FDN'] = instance.isFittingDataNormalized;
  val['TN'] = instance.targetName;
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['CS'] = instance.coefficients.toJson();
  writeNotNull('CPI', instance.costPerIteration);
  writeNotNull('DT', const DTypeJsonConverter().toJson(instance.dtype));
  writeNotNull(r'$V', instance.schemaVersion);
  return val;
}
