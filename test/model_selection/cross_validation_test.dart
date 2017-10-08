import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/model_selection/cross_validator.dart';
import 'package:matcher/matcher.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

const int NUMBER_OF_SAMPLES = 12;

void main() {
  List<Float32x4Vector> features;
  List<double> labels;
  SGDRegressor predictor;

  setUp(() {
    features = new List<Float32x4Vector>.generate(NUMBER_OF_SAMPLES, (_) => new Float32x4Vector.randomFilled(4));
    labels = new Float32x4Vector.randomFilled(NUMBER_OF_SAMPLES).asList();
    predictor = new SGDRegressor();
  });

  group('K-fold cross validator', () {
    test('should return scores vector with proper length', () {
      CrossValidator validator = new CrossValidator.KFold(numberOfFolds: 10);
      Float32x4Vector score2 = validator.evaluate(predictor, features, labels);
      expect(score2.length, equals(10));
    });

    test('should return scores vector with proper length (if `numberOfFolds` argument wasn\'t passed)', () {
      CrossValidator validator = new CrossValidator.KFold();
      Float32x4Vector score = validator.evaluate(predictor, features, labels);
      expect(score.length, equals(5));
    });
  }, skip: true);

  group('LPO cross validator', () {
    test('should return scores vector with proper length', () {
      CrossValidator validator = new CrossValidator.LPO(p: 3);
      Float32x4Vector score = validator.evaluate(predictor, features, labels);
      expect(score.length, equals(220));
    });

    test('should return scores vector with proper length (if `p` argument wasn\'t passed)', () {
      CrossValidator validator = new CrossValidator.LPO();
      Float32x4Vector score = validator.evaluate(predictor, features, labels);
      expect(score.length, equals(792));
    });
  }, skip: true);
}