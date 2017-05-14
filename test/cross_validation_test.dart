import 'package:dart_ml/src/math/vector/typed_vector.dart';
import 'package:dart_ml/src/validators/kfold_cross_validator.dart';
import 'package:dart_ml/src/validators/loo_cross_validator.dart';
import 'package:dart_ml/src/predictors/mbgd_linear_regressor.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  group('Cross validators test.\n', () {
    List<TypedVector> features = new List<TypedVector>.generate(45, (_) => new TypedVector.randomFilled(4));
    TypedVector labels = new TypedVector.randomFilled(45);
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

      TypedVector score1 = validator1.validate(predictor, features, labels);
      TypedVector score2 = validator2.validate(predictor, features, labels);

      expect(score1.length, equals(5));
      expect(score2.length, equals(10));
    });

    test('leave one out validation test: ', () {
      LooCrossValidator validator = new LooCrossValidator();
      TypedVector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(features.length));
    });
  });
}