import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/model_selection/validator/implementation/kfold_cross_validator.dart';
import 'package:dart_ml/src/model_selection/validator/implementation/lpo_cross_validator.dart';
import 'package:dart_ml/src/predictor/gradient_linear_regressor.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

const int NUMBER_OF_SAMPLES = 12;

void main() {
  group('Cross validators test.\n', () {
    List<Vector> features = new List<Vector>.generate(NUMBER_OF_SAMPLES, (_) => new Vector.randomFilled(4));
    Vector labels = new Vector.randomFilled(NUMBER_OF_SAMPLES);
    GradientLinearRegressor predictor;

    setUp(() {
      predictor = new GradientLinearRegressor();
    });

    tearDown(() {
      predictor = null;
    });

    test('k-fold cross validation test: ', () {
      KFoldCrossValidatorImpl validator1 = new KFoldCrossValidatorImpl();
      KFoldCrossValidatorImpl validator2 = new KFoldCrossValidatorImpl(numberOfFolds: 10);

      Vector score1 = validator1.validate(predictor, features, labels);
      Vector score2 = validator2.validate(predictor, features, labels);

      expect(score1.length, equals(5));
      expect(score2.length, equals(10));
    });

    test('leave p out validation test: ', () {
      LpoCrossValidatorImpl validator1 = new LpoCrossValidatorImpl();
      LpoCrossValidatorImpl validator2 = new LpoCrossValidatorImpl(p: 3);

      Vector score1 = validator1.validate(predictor, features, labels);
      Vector score2 = validator2.validate(predictor, features, labels);

      expect(score1.length, equals(792));
      expect(score2.length, equals(220));
    });
  });
}