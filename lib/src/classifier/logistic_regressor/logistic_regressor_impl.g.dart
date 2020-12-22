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
      'CPI'
    ]);
    final val = LogisticRegressorImpl(
      $checkedConvert(
          json,
          'O',
          (v) =>
              const LinearOptimizerTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'I', (v) => v as int),
      $checkedConvert(json, 'LR', (v) => (v as num)?.toDouble()),
      $checkedConvert(json, 'U', (v) => (v as num)?.toDouble()),
      $checkedConvert(json, 'L', (v) => (v as num)?.toDouble()),
      $checkedConvert(json, 'R',
          (v) => const RegularizationTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(json, 'RS', (v) => v as int),
      $checkedConvert(json, 'B', (v) => v as int),
      $checkedConvert(json, 'N', (v) => v as bool),
      $checkedConvert(json, 'LRT',
          (v) => const LearningRateTypeJsonConverter().fromJson(v as String)),
      $checkedConvert(
          json,
          'ICT',
          (v) => const InitialCoefficientsTypeJsonConverter()
              .fromJson(v as String)),
      $checkedConvert(
          json,
          'IC',
          (v) =>
              const VectorJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'CN', (v) => (v as List)?.map((e) => e as String)),
      $checkedConvert(json, 'LF', (v) => fromLinkFunctionJson(v as String)),
      $checkedConvert(json, 'FI', (v) => v as bool),
      $checkedConvert(json, 'IS', (v) => v as num),
      $checkedConvert(
          json,
          'CBC',
          (v) =>
              const MatrixJsonConverter().fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'PT', (v) => v as num),
      $checkedConvert(json, 'NL', (v) => v as num),
      $checkedConvert(json, 'PL', (v) => v as num),
      $checkedConvert(
          json, 'CPI', (v) => (v as List)?.map((e) => e as num)?.toList()),
      $checkedConvert(
          json, 'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
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
    'dtype': 'DT'
  });
}

Map<String, dynamic> _$LogisticRegressorImplToJson(
    LogisticRegressorImpl instance) {
  final val = <String, dynamic>{
    'O':
        const LinearOptimizerTypeJsonConverter().toJson(instance.optimizerType),
  };

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('I', instance.iterationsLimit);
  writeNotNull('LR', instance.initialLearningRate);
  writeNotNull('U', instance.minCoefficientsUpdate);
  writeNotNull('L', instance.lambda);
  writeNotNull(
      'R',
      const RegularizationTypeJsonConverter()
          .toJson(instance.regularizationType));
  writeNotNull('RS', instance.randomSeed);
  val['B'] = instance.batchSize;
  val['N'] = instance.isFittingDataNormalized;
  val['LRT'] =
      const LearningRateTypeJsonConverter().toJson(instance.learningRateType);
  writeNotNull(
      'ICT',
      const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType));
  writeNotNull(
      'IC', const VectorJsonConverter().toJson(instance.initialCoefficients));
  val['CBC'] =
      const MatrixJsonConverter().toJson(instance.coefficientsByClasses);
  val['CN'] = instance.targetNames?.toList();
  val['FI'] = instance.fitIntercept;
  val['IS'] = instance.interceptScale;
  val['DT'] = const DTypeJsonConverter().toJson(instance.dtype);
  val['PT'] = instance.probabilityThreshold;
  val['PL'] = instance.positiveLabel;
  val['NL'] = instance.negativeLabel;
  val['LF'] = linkFunctionToJson(instance.linkFunction);
  writeNotNull('CPI', instance.costPerIteration);
  return val;
}
