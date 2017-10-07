part of 'package:dart_ml/src/implementation.dart';

class RegressionMetricFactory {
  static RegressionMetric RMSE() => const _RMSEMetric();
  static RegressionMetric MAPE() => const _MAPEMetric();
}