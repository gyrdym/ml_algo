import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/common/exception/invalid_class_labels_exception.dart';
import 'package:ml_algo/src/common/exception/invalid_probability_threshold_exception.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../mocks.dart';

void main() {
  group('LogisticRegressorImpl', () {
    final linkFunctionMock = LinkFunctionMock();
    final className = 'class 1';
    final fitIntercept = true;
    final interceptScale = 10.0;
    final coefficients = Matrix.column([1, 2, 3, 4]);
    final probabilityThreshold = 0.6;
    final negativeLabel = -1;
    final positiveLabel = 1;
    final dtype = DType.float32;
    final costPerIteration = [1.2, 233.01, 23, -10001];

    final regressor = LogisticRegressorImpl(
      [className],
      linkFunctionMock,
      fitIntercept,
      interceptScale,
      coefficients,
      probabilityThreshold,
      negativeLabel,
      positiveLabel,
      costPerIteration,
      dtype,
    );

    final testFeatureMatrix = Matrix.fromList([
      [20, 30, 40],
      [20, 22, 11],
      [90, 87, 52],
      [12, 20, 21],
      [33, 44, 55],
    ]);

    final featuresWithIntercept = Matrix.fromList([
      [interceptScale, 20, 30, 40],
      [interceptScale, 20, 22, 11],
      [interceptScale, 90, 87, 52],
      [interceptScale, 12, 20, 21],
      [interceptScale, 33, 44, 55],
    ]);

    final mockedProbabilities = Matrix.column([0.8, 0.5, 0.6, 0.7, 0.3]);

    setUp(() {
      when(linkFunctionMock.link(any)).thenReturn(mockedProbabilities);
    });

    tearDown(() {
      reset(linkFunctionMock);
      injector.clearAll();
      logisticRegressorInjector.clearAll();
    });

    group('default constructor', () {
      test('should create the instance with `classNames` list of just one '
          'element', () {
        expect(regressor.classNames, equals([className]));
      });

      test('should throw an exception if probability threshold is less '
          'than 0', () {
        final probabilityThreshold = -0.1;
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is less '
          'equal to 0', () {
        final probabilityThreshold = 0;
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is equal '
          'to 1', () {
        final probabilityThreshold = 1;
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is greater '
          'than 1', () {
        final probabilityThreshold = 1.2;
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if positive class label equals to '
          'negative class label', () {
        final negativeLabel = 1000;
        final positiveLabel = 1000;
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsA(isA<InvalidClassLabelsException>()));
      });

      test('should throw an exception if no coefficients are provided', () {
        final coefficients = Matrix.empty();
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if coefficients provided for more than '
          'one class', () {
        final coefficients = Matrix.fromList([
          [1, 2, 3],
          [7, 5, 8],
          [2, 0, 9],
          [1, 3, 3],
        ]);
        final actual = () => LogisticRegressorImpl(
          [className],
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          coefficients,
          probabilityThreshold,
          negativeLabel,
          positiveLabel,
          costPerIteration,
          dtype,
        );

        expect(actual, throwsException);
      });

      test('should persist cost per iteration', () {
        expect(regressor.costPerIteration, costPerIteration);
      });
    });

    group('LogisticRegressorImpl.predict', () {
      test('should throw an exception if no test features are provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.empty());

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should consider intercept term', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        regressor.predict(testFeatures);

        verify(linkFunctionMock.link(featuresWithIntercept * coefficients))
            .called(1);
      });

      test('should throw an error if too many features are provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [10, 20, 30, 40],
        ]);
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should throw an error if too few features are provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [10, 20],
        ]);
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should predict class labels basing on calculated probabilities', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);
        final prediction = regressor.predict(testFeatures);

        expect(prediction.rows, equals([
          [positiveLabel],
          [negativeLabel],
          [positiveLabel],
          [positiveLabel],
          [negativeLabel],
        ]));
      });

      test('should return a dataframe with a proper header', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);
        final prediction = regressor.predict(testFeatures);

        expect(prediction.header, equals([className]));
      });
    });
  });
}
