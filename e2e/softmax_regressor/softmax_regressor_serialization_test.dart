import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('SoftmaxRegressor', () {
    test('should deserialize v1 schema version', () async {
      final file = File('e2e/softmax_regressor/softmax_regressor_v1.json');
      final encodedData = await file.readAsString();
      final regressor = SoftmaxRegressor.fromJson(encodedData);

      expect(regressor.targetNames,
          ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']);
      expect(regressor.initialCoefficientsType, InitialCoefficientsType.zeroes);
      expect(regressor.positiveLabel, 1);
      expect(regressor.negativeLabel, 0);
      expect(regressor.iterationsLimit, 100);
      expect(regressor.initialCoefficients, isNull);
      expect(regressor.learningRateType, LearningRateType.constant);
      expect(regressor.optimizerType, LinearOptimizerType.gradient);
      expect(regressor.isFittingDataNormalized, false);
      expect(regressor.batchSize, 1);
      expect(regressor.randomSeed, isNull);
      expect(regressor.lambda, 0);
      expect(regressor.minCoefficientsUpdate, 1e-12);
      expect(regressor.initialLearningRate, 1e-3);
      expect(regressor.decay, 1);
      expect(regressor.regularizationType, isNull);
      expect(regressor.interceptScale, 1.0);
      expect(regressor.fitIntercept, false);
      expect(regressor.dtype, DType.float32);
      expect(regressor.schemaVersion, 3);
    });
  });
}
