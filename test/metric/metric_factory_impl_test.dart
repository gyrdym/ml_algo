import 'package:ml_algo/src/metric/classification/accuracy.dart';
import 'package:ml_algo/src/metric/classification/precision.dart';
import 'package:ml_algo/src/metric/classification/recall.dart';
import 'package:ml_algo/src/metric/metric_factory_impl.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_algo/src/metric/regression/rmse.dart';
import 'package:test/test.dart';

void main() {
  group('MetricFactoryImpl', () {
    const factory = MetricFactoryImpl();

    test('should create RmseMetric instance', () {
      expect(factory.createByType(MetricType.rmse), isA<RmseMetric>());
    });

    test('should create MapeMetric instance', () {
      expect(factory.createByType(MetricType.mape), isA<MapeMetric>());
    });

    test('should create AccuracyMetric instance', () {
      expect(factory.createByType(MetricType.accuracy), isA<AccuracyMetric>());
    });

    test('should create PrecisionMetric instance', () {
      expect(factory.createByType(MetricType.precision), isA<PrecisionMetric>());
    });

    test('should create RecallMetric instance', () {
      expect(factory.createByType(MetricType.recall), isA<RecallMetric>());
    });
  });
}
