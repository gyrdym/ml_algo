import 'dart:io';

import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('LinearRegressor', () {
    test('should deserialize v0 schema version', () async {
      final file = File('e2e/linear_regressor/linear_regressor_v0.json');
      final encodedData = await file.readAsString();
      final regressor = LinearRegressor.fromJson(encodedData);

      expect(regressor.targetName, 'col_13');
      expect(regressor.initialCoefficientsType, InitialCoefficientsType.zeroes);
      expect(regressor.initialCoefficients, isNull);
      expect(regressor.learningRateType, LearningRateType.constant);
      expect(regressor.iterationsLimit, 100);
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
      expect(regressor.coefficients, [
        -0.0006385557935573161,
        0.002575760008767247,
        -0.0005521384882740676,
        -0.000002480079501765431,
        0.00006806042802054435,
        0.001419438631273806,
        -0.006311072036623955,
        0.0014231938403099775,
        -0.0016250074841082096,
        -0.006484318058937788,
        0.0022983804810792208,
        0.0738188698887825,
        -0.004393403884023428
      ]);
    });
  });
}
