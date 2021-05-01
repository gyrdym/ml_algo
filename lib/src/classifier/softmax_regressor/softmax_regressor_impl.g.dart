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
          json,
          'OT',
          (v) =>
              const LinearOptimizerTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'IL', (v) => v as int),
      $checkedConvert(json, 'ILR', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'MCU', (v) => (v as num).toDouble()),
      $checkedConvert(json, 'L', (v) => (v as num).toDouble()),
      $checkedConvert(
          json,
          'RT',
          (v) => const RegularizationTypeJsonConverterNullable()
              .fromJson(v as String?)),
      $checkedConvert(json, 'RS', (v) => v as int?),
      $checkedConvert(json, 'BS', (v) => v as int),
      $checkedConvert(json, 'FDN', (v) => v as bool),
      $checkedConvert(json, 'LR',
          (v) => const LearningRateTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(
          json,
          'ICT',
          (v) => const InitialCoefficientsTypeJsonConverter()
              .fromJson(v as String)),
      $checkedConvert(json, 'IC',
          (v) => v == null ? null : Matrix.fromJson(v as Map<String, dynamic>)),
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
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int?),
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
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('OT',
      const LinearOptimizerTypeJsonConverter().toJson(instance.optimizerType));
  val['IL'] = instance.iterationsLimit;
  val['ILR'] = instance.initialLearningRate;
  val['MCU'] = instance.minCoefficientsUpdate;
  val['L'] = instance.lambda;
  writeNotNull(
      'RT',
      const RegularizationTypeJsonConverterNullable()
          .toJson(instance.regularizationType));
  writeNotNull('RS', instance.randomSeed);
  val['BS'] = instance.batchSize;
  val['FDN'] = instance.isFittingDataNormalized;
  writeNotNull('LR',
      const LearningRateTypeJsonConverter().toJson(instance.learningRateType));
  writeNotNull(
      'ICT',
      const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType));
  writeNotNull('IC', instance.initialCoefficients?.toJson());
  val['CN'] = instance.targetNames.toList();
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['CBC'] = instance.coefficientsByClasses.toJson();
  writeNotNull('DT', const DTypeJsonConverter().toJson(instance.dtype));
  writeNotNull('LF', linkFunctionToJson(instance.linkFunction));
  val['PL'] = instance.positiveLabel;
  val['NL'] = instance.negativeLabel;
  writeNotNull('CPI', instance.costPerIteration);
  writeNotNull(r'$V', instance.schemaVersion);
  return val;
}
