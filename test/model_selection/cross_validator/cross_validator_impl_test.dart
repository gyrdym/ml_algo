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
    test('should perform validation of a model on given test indices of'
        'observations', () {
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

    test('should use returned data (first element - training data, second '
        'element - testing data) from the callback for data preprocessing '
        'while evaluating a predictor', () {
      final allObservations = DataFrame(
        <Iterable<num>>[
          [1, 1, 1, 1],
        ],
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

      final dataPreprocessFnResponse = <int, List<DataFrame>>{
        0: [
          DataFrame(<Iterable<num>>[
            [1, 2, 3, 4],
            [7, 8, 9, 0],
          ], headerExists: false),
          DataFrame(<Iterable<num>>[
            [14, 50, 39, 24],
            [77, 38, 29, 70],
          ], headerExists: false),
        ],
        1: [
          DataFrame(<Iterable<num>>[
            [100, 200, 300, 400],
            [117, 118, 119, 110],
          ], headerExists: false),
          DataFrame(<Iterable<num>>[
            [154, 550, 939, 124],
          ], headerExists: false),
        ],
        2: [
          DataFrame(<Iterable<num>>[
            [700, 500, 900, 600],
            [111, 888, 999, 222],
            [301, 403, 501, 607],
          ], headerExists: false),
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
        dataPreprocessFn: (trainData, testData) =>
          dataPreprocessFnResponse[iterationCounter++],
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
  });
}
