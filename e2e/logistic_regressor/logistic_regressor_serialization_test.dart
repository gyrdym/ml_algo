import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('LogisticRegressor', () {
    test('should deserialize v1 schema version', () async {
      final file = File('e2e/logistic_regressor/logistic_regressor_v1.json');
      final encodedData = await file.readAsString();
      final regressor = LogisticRegressor.fromJson(encodedData);

      expect(regressor.targetNames, ['class variable (0 or 1)']);
      expect(regressor.initialCoefficientsType, InitialCoefficientsType.zeroes);
      expect(regressor.positiveLabel, 1);
      expect(regressor.negativeLabel, 0);
      expect(regressor.probabilityThreshold, 0.5);
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
      expect(regressor.regularizationType, isNull);
      expect(regressor.interceptScale, 1.0);
      expect(regressor.fitIntercept, false);
      expect(regressor.dtype, DType.float32);
      expect(regressor.schemaVersion, 3);
    });
  });
}
