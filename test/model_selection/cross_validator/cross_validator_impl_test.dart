import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

Splitter createSplitter(Iterable<Iterable<int>> indices) {
  final splitter = SplitterMock();
  when(splitter.split(any)).thenReturn(indices);
  return splitter;
}

void main() {
  group('CrossValidatorImpl', () {
    test('should perform validation of a predictor on given test splits', () {
      final allObservations = DataFrame(<Iterable<num>>[
        [330, 930, 130, 100],
        [630, 830, 230, 200],
        [730, 730, 330, 300],
        [830, 630, 430, 400],
        [930, 530, 530, 500],
        [130, 430, 630, 600],
        [230, 330, 730, 700],
        [430, 230, 830, 800],
        [530, 130, 930, 900],
      ], header: ['first', 'second', 'third', 'target'], headerExists: false);

      final metric = MetricType.mape;
      final splitter = createSplitter([[0,2,4],[6, 8]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(allObservations,
          ['target'], splitter, DType.float32);

      var score = 20.0;
      when(predictor.assess(any, any, any))
          .thenAnswer((Invocation inv) => score = score + 10);

      final actual = validator
          .evaluate((observations, outcomes) => predictor, metric);

      expect(actual, 35);

      final verificationResult = verify(
        predictor.assess(
          captureThat(isNotNull),
          argThat(equals(['target'])),
          metric,
        ));
      final firstAssessCallArgs = verificationResult.captured;

      expect((firstAssessCallArgs[0] as DataFrame).rows, equals([
        [330, 930, 130, 100],
        [730, 730, 330, 300],
        [930, 530, 530, 500],
      ]));

      expect((firstAssessCallArgs[1] as DataFrame).rows, equals([
        [230, 330, 730, 700],
        [530, 130, 930, 900],
      ]));

      verificationResult.called(2);
    });

    test('should take the first element as train samples from data '
        'preprocessing callback response while evaluating a predictor', () {
      // we don't care about data here cause it will be mocked farther
      final allObservations = DataFrame(
        [<num>[1, 1, 1, 1]],
        header: ['first', 'second', 'third', 'target'],
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [0], [0]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(allObservations,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      int iterationCounter = 0;

      final iterationToResponse = <int, List<DataFrame>>{
        0: [
          DataFrame(<Iterable<num>>[
            [1, 2, 3, 4],
            [7, 8, 9, 0],
          ], headerExists: false),
          DataFrame(<Iterable<num>>[[1, 1, 1, 1]], headerExists: false),
        ],
        1: [
          DataFrame(<Iterable<num>>[
            [100, 200, 300, 400],
            [117, 118, 119, 110],
          ], headerExists: false),
          DataFrame(<Iterable<num>>[[2, 2, 2, 2]], headerExists: false),
        ],
        2: [
          DataFrame(<Iterable<num>>[
            [700, 500, 900, 600],
            [111, 888, 999, 222],
            [301, 403, 501, 607],
          ], headerExists: false),
          DataFrame(<Iterable<num>>[[3, 3, 3, 3]], headerExists: false),
        ],
      };

      validator.evaluate(
        (observations, outcomes) {
          expect(
            observations.toMatrix(),
            equals(iterationToResponse[iterationCounter++][0].toMatrix()),
          );
          return predictor;
        },
        metric,
        onDataSplit: (trainData, testData) =>
          iterationToResponse[iterationCounter],
      );
    });

    test('should take the second element as test samples from data '
        'preprocessing callback response while evaluating a predictor', () {
      // we don't care about data here cause it will be mocked farther
      final allObservations = DataFrame(
        [<num>[1, 1, 1, 1]],
        header: ['first', 'second', 'third', 'target'],
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [0], [0]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(allObservations,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      int iterationCounter = 0;

      final iterationToResponse = <int, List<DataFrame>>{
        0: [
          DataFrame(<Iterable<num>>[[1, 1, 1, 1]], headerExists: false),
          DataFrame(<Iterable<num>>[
            [14, 50, 39, 24],
            [77, 38, 29, 70],
          ], headerExists: false),
        ],
        1: [
          DataFrame(<Iterable<num>>[[2, 2, 2, 2]], headerExists: false),
          DataFrame(<Iterable<num>>[
            [154, 550, 939, 124],
          ], headerExists: false),
        ],
        2: [
          DataFrame(<Iterable<num>>[[3, 3, 3, 3]], headerExists: false),
          DataFrame(<Iterable<num>>[
            [44, 55, 66, 11],
            [29, 22, 11,  0],
            [91, 32, 16, 17],
          ], headerExists: false),
        ],
      };

      validator.evaluate(
        (observations, outcomes) => predictor,
        metric,
        onDataSplit: (trainData, testData) =>
        iterationToResponse[iterationCounter++],
      );

      final verificationResult = verify(
          predictor.assess(
            captureThat(isNotNull),
            argThat(equals(['target'])),
            metric,
          ));
      final firstAssessCallArgs = verificationResult.captured;

      expect((firstAssessCallArgs[0] as DataFrame).rows, equals([
        [14, 50, 39, 24],
        [77, 38, 29, 70],
      ]));

      expect((firstAssessCallArgs[1] as DataFrame).rows, equals([
        [154, 550, 939, 124],
      ]));

      expect((firstAssessCallArgs[2] as DataFrame).rows, equals([
        [44, 55, 66, 11],
        [29, 22, 11,  0],
        [91, 32, 16, 17],
      ]));

      verificationResult.called(3);
    });

    test('should pass splits into data preprocessing callback', () {
      final header = ['first', 'second', 'third', 'target'];

      // we don't care about data here cause it will be mocked farther
      final allObservations = DataFrame(
        <Iterable<num>>[
          [ 1,  1,  1,   1],
          [ 2,  3,  4,   5],
          [18, 71, 15,  61],
          [19,  0, 21, 331],
          [11, 10,  9,  40],
        ],
        header: header,
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [2], [4]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(allObservations,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      int iterationCounter = 0;

      final iterationToSplits = {
        0: {
          'trainData': DataFrame(
              <Iterable<num>>[
                [ 2, 3, 4, 5],
                [18, 71, 15, 61],
                [19, 0, 21, 331],
                [11, 10, 9, 40],
              ],
              header: header,
              headerExists: false,
          ),
          'testData': DataFrame(
              [<num>[ 1, 1, 1, 1]],
              header: header,
              headerExists: false,
          ),
        },
        1: {
          'trainData': DataFrame(
              <Iterable<num>>[
                [ 1, 1, 1, 1],
                [ 2, 3, 4, 5],
                [19, 0, 21, 331],
                [11, 10, 9, 40],
              ],
              header: header,
              headerExists: false,
          ),
          'testData': DataFrame(
              [<num>[18, 71, 15, 61]],
              header: header,
              headerExists: false,
          ),
        },
        2: {
          'trainData': DataFrame(
              <Iterable<num>>[
                [ 1, 1, 1, 1],
                [ 2, 3, 4, 5],
                [18, 71, 15, 61],
                [19, 0, 21, 331],
              ],
              header: header,
              headerExists: false,
          ),
          'testData': DataFrame(
              [<num>[11, 10, 9, 40]],
              header: header,
              headerExists: false,
          ),
        },
      };

      validator.evaluate(
        (observations, outcomes) => predictor,
        metric,
        onDataSplit: (trainData, testData) {
          final expectedSplits = iterationToSplits[iterationCounter++];

          expect(trainData.header, expectedSplits['trainData'].header);
          expect(trainData.rows, equals(expectedSplits['trainData'].rows));

          expect(testData.header, expectedSplits['testData'].header);
          expect(testData.rows, equals(expectedSplits['testData'].rows));

          return [
            DataFrame([<num>[1, 2, 3, 4]], headerExists: false),
            DataFrame([<num>[0, 0, 0, 0]], headerExists: false),
          ];
        }
      );
    });

    test('should throw an exception if one tries to return a training data '
        'from data perprocessing callback with number of columns less than '
        'the number of columns of original data', () {
      final header = ['first', 'second', 'third', 'target'];

      // we don't care about data here cause it will be mocked farther
      final originalData = DataFrame(
        <Iterable<num>>[
          [ 1,  1,  1,   1],
          [ 2,  3,  4,   5],
          [18, 71, 15,  61],
          [19,  0, 21, 331],
          [11, 10,  9,  40],
        ],
        header: header,
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [2], [4]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(originalData,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      final transformedTrainData = DataFrame([<num>[1, 2]],
          headerExists: false);
      final transformedTestData = DataFrame([<num>[0, 0, 0, 0]],
          headerExists: false);

      final actual = () => validator.evaluate(
              (observations, outcomes) => predictor,
          metric,
          onDataSplit: (trainData, testData) =>
            [
              transformedTrainData,
              transformedTestData,
            ],
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if one tries to return a training data '
        'from data perprocessing callback with number of columns greater than '
        'the number of columns of original data', () {
      final header = ['first', 'second', 'third', 'target'];

      // we don't care about data here cause it will be mocked farther
      final originalData = DataFrame(
        <Iterable<num>>[
          [ 1,  1,  1,   1],
          [ 2,  3,  4,   5],
          [18, 71, 15,  61],
          [19,  0, 21, 331],
          [11, 10,  9,  40],
        ],
        header: header,
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [2], [4]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(originalData,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      final transformedTrainData = DataFrame([<num>[1, 2, 3, 4, 5]],
          headerExists: false);
      final transformedTestData = DataFrame([<num>[0, 0, 0, 0]],
          headerExists: false);

      final actual = () => validator.evaluate(
        (observations, outcomes) => predictor,
        metric,
        onDataSplit: (trainData, testData) => [
          transformedTrainData,
          transformedTestData,
        ],
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if one tries to return a testing data '
        'from data perprocessing callback with number of columns less than '
        'the number of columns of original data', () {
      final header = ['first', 'second', 'third', 'target'];

      // we don't care about data here cause it will be mocked farther
      final originalData = DataFrame(
        <Iterable<num>>[
          [ 1,  1,  1,   1],
          [ 2,  3,  4,   5],
          [18, 71, 15,  61],
          [19,  0, 21, 331],
          [11, 10,  9,  40],
        ],
        header: header,
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [2], [4]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(originalData,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      final transformedTrainData = DataFrame([<num>[1, 2, 3, 4]],
          headerExists: false);
      final transformedTestData = DataFrame([<num>[0, 0, 0]],
          headerExists: false);

      final actual = () => validator.evaluate(
        (observations, outcomes) => predictor,
        metric,
        onDataSplit: (trainData, testData) => [
          transformedTrainData,
          transformedTestData,
        ],
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if one tries to return a testing data '
        'from data perprocessing callback with number of columns greater than '
        'the number of columns of original data', () {
      final header = ['first', 'second', 'third', 'target'];

      // we don't care about data here cause it will be mocked farther
      final originalData = DataFrame(
        <Iterable<num>>[
          [ 1,  1,  1,   1],
          [ 2,  3,  4,   5],
          [18, 71, 15,  61],
          [19,  0, 21, 331],
          [11, 10,  9,  40],
        ],
        header: header,
        headerExists: false,
      );

      final metric = MetricType.mape;
      final splitter = createSplitter([[0], [2], [4]]);
      final predictor = AssessableMock();
      final validator = CrossValidatorImpl(originalData,
          ['target'], splitter, DType.float32);

      when(predictor.assess(any, any, any)).thenReturn(1);

      final transformedTrainData = DataFrame([<num>[1, 2, 3, 4]],
          headerExists: false);
      final transformedTestData = DataFrame([<num>[0, 0, 0, 0, 0]],
          headerExists: false);

      final actual = () => validator.evaluate(
        (observations, outcomes) => predictor,
        metric,
        onDataSplit: (trainData, testData) => [
          transformedTrainData,
          transformedTestData,
        ],
      );

      expect(actual, throwsException);
    });
  });
}
