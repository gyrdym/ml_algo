import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Logistic regressor json key', () {
    test('should have a proper json key for coefficient by classes field', () {
      expect(logisticRegressorCoefficientsByClassesJsonKey, 'CBC');
    });

    test('should have a proper json key for class names json field', () {
      expect(logisticRegressorClassNamesJsonKey, 'CN');
    });

    test('should have a proper json key for fit intercept json field', () {
      expect(logisticRegressorFitInterceptJsonKey, 'FI');
    });

    test('should have a proper json key for intercept scale json field', () {
      expect(logisticRegressorInterceptScaleJsonKey, 'IS');
    });

    test('should have a proper json key for dtype json field', () {
      expect(logisticRegressorDTypeJsonKey, 'DT');
    });

    test('should have a proper json key for probability threshold json '
        'field', () {
      expect(logisticRegressorProbabilityThresholdJsonKey, 'PT');
    });

    test('should have a proper json key for positive label json field', () {
      expect(logisticRegressorPositiveLabelJsonKey, 'PL');
    });

    test('should have a proper json key for negative label json field', () {
      expect(logisticRegressorNegativeLabelJsonKey, 'NL');
    });

    test('should have a proper json key for link function json field', () {
      expect(logisticRegressorLinkFunctionJsonKey, 'LF');
    });
  });
}
