import 'package:di/di.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/math.dart';
import 'package:dart_ml/src/math/math_impl.dart';
import 'package:dart_ml/src/model_selection/validator/validator_impl.dart';
import 'package:dart_ml/src/data_splitter/data_splitter.dart';
import 'package:dart_ml/src/data_splitter/data_splitter_impl.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart' show SGDOptimizer, GradientOptimizerType;
import 'package:dart_ml/src/optimizer/optimizer_impl.dart' show SGDOptimizerImpl;
import 'package:dart_ml/src/predictor/linear_regressor/gradient/implementation/gradient_linear_regressor.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

const int NUMBER_OF_SAMPLES = 12;

void main() {
  List<Vector> features;
  Vector labels;
  GradientLinearRegressor predictor;

  setUp(() {
    injector = new ModuleInjector([
      new Module()
        ..bind(Randomizer, toFactory: () => new RandomizerImpl())
        ..bind(KFoldSplitter, toFactory: () => new KFoldSplitterImpl())
        ..bind(LeavePOutSplitter, toFactory: () => new LeavePOutSplitterImpl())
        ..bind(SGDOptimizer, toFactory: () => new SGDOptimizerImpl())
    ]);

    features = new List<Vector>.generate(NUMBER_OF_SAMPLES, (_) => new Vector.randomFilled(4));
    labels = new Vector.randomFilled(NUMBER_OF_SAMPLES);
    predictor = new GradientLinearRegressor(optimizerType: GradientOptimizerType.SGD);
  });

  group('K-fold cross validator', () {
    test('should return scores vector with proper length', () {
      KFoldCrossValidatorImpl validator = new KFoldCrossValidatorImpl(numberOfFolds: 10);
      Vector score2 = validator.validate(predictor, features, labels);
      expect(score2.length, equals(10));
    });

    test('should return scores vector with proper length (if `numberOfFolds` argument wasn\'t passed)', () {
      KFoldCrossValidatorImpl validator = new KFoldCrossValidatorImpl();
      Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(5));
    });
  });

  group('LPO cross validator', () {
    test('should return scores vector with proper length', () {
      LpoCrossValidatorImpl validator = new LpoCrossValidatorImpl(p: 3);
      Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(220));
    });

    test('should return scores vector with proper length', () {
      LpoCrossValidatorImpl validator = new LpoCrossValidatorImpl();
      Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(792));
    });
  });
}