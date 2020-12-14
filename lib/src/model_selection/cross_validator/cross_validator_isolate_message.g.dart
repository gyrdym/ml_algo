// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'cross_validator_isolate_message.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

CrossValidatorIsolateMessage _$CrossValidatorIsolateMessageFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('CrossValidatorIsolateMessage', json, () {
    $checkKeys(json, allowedKeys: const ['P', 'T', 'TE', 'M']);
    final val = CrossValidatorIsolateMessage(
      $checkedConvert(json, 'P',
          (v) => fromSerializablePredictorJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json,
          'T',
          (v) =>
              v == null ? null : DataFrame.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json,
          'TE',
          (v) =>
              v == null ? null : DataFrame.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(
          json, 'M', (v) => _$enumDecodeNullable(_$MetricTypeEnumMap, v)),
    );
    return val;
  }, fieldKeyMap: const {
    'predictorPrototype': 'P',
    'trainData': 'T',
    'testData': 'TE',
    'metricType': 'M'
  });
}

Map<String, dynamic> _$CrossValidatorIsolateMessageToJson(
        CrossValidatorIsolateMessage instance) =>
    <String, dynamic>{
      'P': instance.predictorPrototype,
      'T': instance.trainData,
      'TE': instance.testData,
      'M': _$MetricTypeEnumMap[instance.metricType],
    };

T _$enumDecode<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    throw ArgumentError('A value must be provided. Supported values: '
        '${enumValues.values.join(', ')}');
  }

  final value = enumValues.entries
      .singleWhere((e) => e.value == source, orElse: () => null)
      ?.key;

  if (value == null && unknownValue == null) {
    throw ArgumentError('`$source` is not one of the supported values: '
        '${enumValues.values.join(', ')}');
  }
  return value ?? unknownValue;
}

T _$enumDecodeNullable<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    return null;
  }
  return _$enumDecode<T>(enumValues, source, unknownValue: unknownValue);
}

const _$MetricTypeEnumMap = {
  MetricType.mape: 'mape',
  MetricType.rmse: 'rmse',
  MetricType.rss: 'rss',
  MetricType.accuracy: 'accuracy',
  MetricType.precision: 'precision',
  MetricType.recall: 'recall',
};
