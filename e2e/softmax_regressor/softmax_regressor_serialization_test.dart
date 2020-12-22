import 'dart:io';

import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('SoftmaxRegressor', () {
    test('should deserialize v0 schema version', () async {
      final file = File('e2e/softmax_regressor/softmax_regressor_v0.json');
      final encodedData = await file.readAsString();
      final regressor = SoftmaxRegressor.fromJson(encodedData);

      expect(regressor.targetNames,
          ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']);
      expect(regressor.initialCoefficientsType, isNull);
      expect(regressor.positiveLabel, 1);
      expect(regressor.negativeLabel, 0);
      expect(regressor.iterationsLimit, isNull);
      expect(regressor.initialCoefficients, isNull);
      expect(regressor.learningRateType, isNull);
      expect(regressor.optimizerType, isNull);
      expect(regressor.isFittingDataNormalized, isNull);
      expect(regressor.batchSize, isNull);
      expect(regressor.randomSeed, isNull);
      expect(regressor.lambda, isNull);
      expect(regressor.minCoefficientsUpdate, isNull);
      expect(regressor.initialLearningRate, isNull);
      expect(regressor.regularizationType, isNull);
      expect(regressor.interceptScale, 1.0);
      expect(regressor.fitIntercept, false);
      expect(regressor.dtype, DType.float32);
    });
  });
}
