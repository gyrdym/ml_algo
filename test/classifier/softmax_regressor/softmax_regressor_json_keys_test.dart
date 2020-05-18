import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('Softmax regressor json keys', () {
    test('should contain a proper key for class names field', () {
      expect(softmaxRegressorClassNamesJsonKey, 'CN');
    });

    test('should contain a proper key for fit intercept field', () {
      expect(softmaxRegressorFitInterceptJsonKey, 'FI');
    });

    test('should contain a proper key for intercept scale field', () {
      expect(softmaxRegressorInterceptScaleJsonKey, 'IS');
    });

    test('should contain a proper key for coefficients by classes field', () {
      expect(softmaxRegressorCoefficientsByClassesJsonKey, 'CBC');
    });

    test('should contain a proper key for dtype field', () {
      expect(softmaxRegressorDTypeJsonKey, 'DT');
    });

    test('should contain a proper key for link function feld', () {
      expect(softmaxRegressorLinkFunctionJsonKey, 'LF');
    });

    test('should contain a proper key for positive label field', () {
      expect(softmaxRegressorPositiveLabelJsonKey, 'PL');
    });

    test('should contain a proper key for negative label field', () {
      expect(softmaxRegressorNegativeLabelJsonKey, 'NL');
    });
  });
}
