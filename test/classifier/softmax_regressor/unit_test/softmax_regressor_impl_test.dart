import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../mocks.dart';

void main() {
  group('SoftmaxRegressorImpl', () {
    final linkFunctionMock = LinkFunctionMock();
    final fitIntercept = true;
    final interceptScale = 10;
    final positiveLabel = 1.0;
    final negativeLabel = -1.0;
    final costPerIteration = [10, -10, 20, 2.3];
    final dtype = DType.float32;

    final coefficientsByClasses = Matrix.fromList([
      [-1, -3,   -5],
      [ 2, 10,  200],
      [ 1, 40,  300],
      [ 3, 44, -120],
      [ 3, 22,   10],
      [ 4,  0,    1],
    ]);

    final testFeatureMatrix = Matrix.fromList([
      [ 10, 20,  30,   40,   50],
      [ 11, 22,  33,   44,   55],
      [ 31, 32,  93,   34,   35],
      [ 91, 72,  83,   74,   65],
      [ 55, 11,  99,   33,   12],
      [-11,  0, 123, 1000, 1e-4],
    ]);

    final interceptColumn = Vector.fromList([
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
    ]);

    final testFeaturesMatrixWithIntercept = Matrix.fromColumns([
      interceptColumn,
      ...testFeatureMatrix.columns,
    ]);

    final mockedProbabilities = Matrix.fromList([
      [ 0.6,  0.2,  0.2],
      [ 0.1,  0.8,  0.1],
      [ 0.1,  0.7,  0.2],
      [0.05, 0.05,  0.9],
      [ 0.5,  0.4,  0.1],
      [0.05,  0.8, 0.15],
    ]);

    final expectedOutcomeMatrix = Matrix.fromList([
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
      [negativeLabel, negativeLabel, positiveLabel],
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
    ]);

    final firstClass = 'target_1';
    final secondClass = 'target_2';
    final thirdClass = 'target_3';
    final targetNames = [firstClass, secondClass, thirdClass];

    final testFeatures = DataFrame.fromMatrix(
      testFeatureMatrix,
      header: ['1', '2', '3', '4', '5', ...targetNames],
    );

    SoftmaxRegressorImpl regressor;

    setUp(() {
      when(linkFunctionMock.link(any)).thenReturn(mockedProbabilities);

      regressor = SoftmaxRegressorImpl(
        coefficientsByClasses,
        targetNames,
        linkFunctionMock,
        fitIntercept,
        interceptScale,
        positiveLabel,
        negativeLabel,
        costPerIteration,
        dtype,
      );
    });

    tearDown(() {
      reset(linkFunctionMock);
      injector.clearAll();
      softmaxRegressorInjector.clearAll();
    });

    group('constructor', () {
      test('should throw an exception if no coefficients are provided', () {
        final actual = () => SoftmaxRegressorImpl(
          Matrix.empty(),
          targetNames,
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          positiveLabel,
          negativeLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if coefficients for too few number of '
          'classes are provided', () {
        final actual = () => SoftmaxRegressorImpl(
          Matrix.fromList([
            [1],
            [2],
            [3],
          ]),
          targetNames,
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          positiveLabel,
          negativeLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsException);
      });
    });

    group('`predict` method', () {
      test('should throw an exception if no features provided', () {
        final testFeatureMatrix = Matrix.empty();
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should throw an exception if too many features provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [1, 2, 3, 4, 5, 6, 7, 8],
        ]);

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should throw an exception if too few features provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [1, 2, 3],
        ]);

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should consider intercept term', () {
        regressor.predict(testFeatures);

        verify(linkFunctionMock.link(
          argThat(iterable2dAlmostEqualTo(
              testFeaturesMatrixWithIntercept * coefficientsByClasses),
          ),
        )).called(1);
      });

      test('should use a class with maximal probability for prediction', () {
        final actual = regressor.predict(testFeatures);

        expect(actual.toMatrix(dtype), equals(expectedOutcomeMatrix));
      });

      test('should predict the first class if outcome is equiprobable', () {
        reset(linkFunctionMock);

        when(linkFunctionMock.link(any)).thenReturn(Matrix.fromList([
          [0.33, 0.33, 0.33],
        ]));
        
        final actual = regressor.predict(testFeatures);
        final expectedOutcome = Matrix.fromList([
          [positiveLabel, negativeLabel, negativeLabel],
        ]);

        expect(actual.toMatrix(dtype), equals(expectedOutcome));
      });

      test('should return a dataframe with a proper header', () {
        final actual = regressor.predict(testFeatures);

        expect(actual.header, equals(targetNames));
      });
    });

    group('`predictProbabilities` method', () {
      test('should throw an exception if no features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.empty());

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should throw an exception if too few features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.fromList([
          [1, 2],
        ]));

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should throw an exception if too many features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.fromList([
          [1, 2, 4, 4, 5, 6],
        ]));

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should consider intercept term', () {
        regressor.predictProbabilities(testFeatures);

        verify(linkFunctionMock.link(
            testFeaturesMatrixWithIntercept * coefficientsByClasses)
        ).called(1);
      });

      test('should return probabilities as dataframe', () {
        final probabilities = regressor.predictProbabilities(testFeatures);

        expect(probabilities.rows, equals(mockedProbabilities));
      });

      test('should return a dataframe with a proper header', () {
        final probabilities = regressor.predictProbabilities(testFeatures);

        expect(probabilities.header, equals(targetNames));
      });

      test('should persist cost per iteration list', () {
        expect(regressor.costPerIteration, costPerIteration);
      });
    });
  });
}
