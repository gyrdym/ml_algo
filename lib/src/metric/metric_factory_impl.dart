import 'package:ml_algo/src/metric/classification/accuracy.dart';
import 'package:ml_algo/src/metric/classification/precision.dart';
import 'package:ml_algo/src/metric/classification/recall.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_algo/src/metric/regression/rmse.dart';

class MetricFactoryImpl implements MetricFactory {
  const MetricFactoryImpl();

  @override
  Metric createByType(MetricType type) {
    switch (type) {
      case MetricType.rmse:
        return const RmseMetric();

      case MetricType.mape:
        return const MapeMetric();

      case MetricType.accuracy:
        return const AccuracyMetric();

      case MetricType.precision:
        return const PrecisionMetric();

      case MetricType.recall:
        return const RecallMetric();

      default:
        throw UnsupportedError('Unsupported metric type $type');
    }
  }
}
