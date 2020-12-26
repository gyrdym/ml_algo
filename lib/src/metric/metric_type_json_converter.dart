import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/metric_type_encoded_values.dart';

class MetricTypeJsonConverter implements JsonConverter<MetricType, String> {
  const MetricTypeJsonConverter();

  @override
  MetricType fromJson(String json) {
    switch (json) {
      case mapeMetricTypeEncodedValue:
        return MetricType.mape;

      case rmseMetricTypeEncodedValue:
        return MetricType.rmse;

      case rssMetricTypeEncodedValue:
        return MetricType.rss;

      case accuracyMetricTypeEncodedValue:
        return MetricType.accuracy;

      case precisionMetricTypeEncodedValue:
        return MetricType.precision;

      case recallMetricTypeEncodedValue:
        return MetricType.recall;

      default:
        throw UnsupportedError('Unsupported encoded metric value - $json');
    }
  }

  @override
  String toJson(MetricType metricType) {
    switch (metricType) {
      case MetricType.mape:
        return mapeMetricTypeEncodedValue;

      case MetricType.rmse:
        return rmseMetricTypeEncodedValue;

      case MetricType.rss:
        return rssMetricTypeEncodedValue;

      case MetricType.accuracy:
        return accuracyMetricTypeEncodedValue;

      case MetricType.precision:
        return precisionMetricTypeEncodedValue;

      case MetricType.recall:
        return recallMetricTypeEncodedValue;

      default:
        throw UnsupportedError('Unsupported metric type - $metricType');
    }
  }
}
