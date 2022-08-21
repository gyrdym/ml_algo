import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

num trainHousingModel(MetricType metricType, DType dtype) {
  final data = getHousingDataFrame().shuffle();
  final samples = splitData(data, [0.8]);
  final trainSamples = samples.first;
  final testSamples = samples.last;
  final targetName = 'MEDV';

  final model = LinearRegressor.newton(
    trainSamples,
    targetName,
    dtype: dtype,
  );

  return model.assess(testSamples, metricType);
}

num trainWineModel(MetricType metricType, DType dtype) {
  final data = getWineQualityDataFrame().shuffle();
  final samples = splitData(data, [0.8]);
  final trainSamples = samples.first;
  final testSamples = samples.last;
  final targetName = 'quality';

  final model = LinearRegressor.newton(
    trainSamples,
    targetName,
    dtype: dtype,
  );

  return model.assess(testSamples, metricType);
}

void main() {
  group('LinearRegressor, Newton method, housing dataset', () {
    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float32', () {
      final error = trainHousingModel(MetricType.mape, DType.float32);

      print('MAPE is $error');

      expect(error, lessThan(0.5));
    });

    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float64', () {
      final error = trainHousingModel(MetricType.mape, DType.float64);

      print('MAPE is $error');

      expect(error, lessThan(0.2));
    });
  });

  group('LinearRegressor, Newton method, wine dataset', () {
    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float32', () {
      final error = trainWineModel(MetricType.mape, DType.float32);

      print('MAPE is $error');

      expect(error, lessThan(0.2));
    });

    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float64', () {
      final error = trainWineModel(MetricType.mape, DType.float64);

      print('MAPE is $error');

      expect(error, lessThan(0.5));
    });
  });
}
