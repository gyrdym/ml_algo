import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/model_selection/model_selection.dart' show CrossValidator;

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

const int NUMBER_OF_SAMPLES = 12;

void main() {
  List<Float32x4Vector> features;
  List<double> labels;
  SGDRegressor predictor;

  setUp(() {
    injector = new ModuleInjector([
      new Module()
        ..bind(Randomizer, toFactory: () => MathUtils.createRandomizer())
        ..bind(InitialWeightsGenerator, toFactory: () => InitialWeightsGeneratorFactory.createZeroWeightsGenerator())
        ..bind(LearningRateGenerator, toFactory: () => LearningRateGeneratorFactory.createSimpleGenerator())
        ..bind(GradientCalculator, toFactory: () => MathUtils.createGradientCalculator())
        ..bind(KFoldSplitter, toFactory: () => DataSplitterFactory.createKFoldSplitter())
        ..bind(LeavePOutSplitter, toFactory: () => DataSplitterFactory.createLpoSplitter())
        ..bind(SGDOptimizer, toFactory: () => GradientOptimizerFactory.createStochasticOptimizer())
    ]);

    features = new List<Float32x4Vector>.generate(NUMBER_OF_SAMPLES, (_) => new Float32x4Vector.randomFilled(4));
    labels = new Float32x4Vector.randomFilled(NUMBER_OF_SAMPLES).asList();
    predictor = new SGDRegressor();
  });

  group('K-fold cross validator', () {
    test('should return scores vector with proper length', () {
      CrossValidator validator = new CrossValidator.KFold(numberOfFolds: 10);
      Float32x4Vector score2 = validator.validate(predictor, features, labels);
      expect(score2.length, equals(10));
    });

    test('should return scores vector with proper length (if `numberOfFolds` argument wasn\'t passed)', () {
      CrossValidator validator = new CrossValidator.KFold();
      Float32x4Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(5));
    });
  });

  group('LPO cross validator', () {
    test('should return scores vector with proper length', () {
      CrossValidator validator = new CrossValidator.LPO(p: 3);
      Float32x4Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(220));
    });

    test('should return scores vector with proper length (if `p` argument wasn\'t passed)', () {
      CrossValidator validator = new CrossValidator.LPO();
      Float32x4Vector score = validator.validate(predictor, features, labels);
      expect(score.length, equals(792));
    });
  });
}