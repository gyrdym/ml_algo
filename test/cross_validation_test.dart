import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/validator/kfold_cross_validator.dart';
import 'package:dart_ml/src/validator/lpo_cross_validator.dart';
import 'package:dart_ml/src/predictor/mbgd_linear_regressor.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

const int NUMBER_OF_SAMPLES = 12;

void main() {
  group('Cross validators test.\n', () {
    List<Vector> features = new List<Vector>.generate(NUMBER_OF_SAMPLES, (_) => new Vector.randomFilled(4));
    Vector labels = new Vector.randomFilled(NUMBER_OF_SAMPLES);
    MBGDLinearRegressor predictor;

    setUp(() {
      predictor = new MBGDLinearRegressor();
    });

    tearDown(() {
      predictor = null;
    });

    test('k-fold cross validation test: ', () {
      KFoldCrossValidator validator1 = new KFoldCrossValidator();
      KFoldCrossValidator validator2 = new KFoldCrossValidator(numberOfFolds: 10);

      Vector score1 = validator1.validate(predictor, features, labels);
      Vector score2 = validator2.validate(predictor, features, labels);

      expect(score1.length, equals(5));
      expect(score2.length, equals(10));
    });

    test('leave p out validation test: ', () {
      LpoCrossValidator validator1 = new LpoCrossValidator();
      LpoCrossValidator validator2 = new LpoCrossValidator(p: 3);

      Vector score1 = validator1.validate(predictor, features, labels);
      Vector score2 = validator2.validate(predictor, features, labels);

      expect(score1.length, equals(792));
      expect(score2.length, equals(220));
    });
  });
}