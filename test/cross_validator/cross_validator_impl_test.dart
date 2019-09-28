import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/mocks.dart';

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
      ], header: ['first', 'second', 'third', 'target']);

      final metric = MetricType.mape;
      final splitter = createSplitter([[0,2,4],[6, 8]]);
      final predictor = PredictorMock();
      final validator = CrossValidatorImpl(allObservations,
          ['target'], splitter, DType.float32);

      var score = 20.0;
      when(predictor.assess(any, any, any))
          .thenAnswer((Invocation inv) => score = score + 10);

      final actual = validator
          .evaluate((observations, outcomes) => predictor, metric);

      expect(actual, 35);

      verify(predictor.assess(argThat(equals([
        [330, 930, 130],
        [730, 730, 330],
        [930, 530, 530],
      ])), argThat(equals([[100], [300], [500]])), metric)).called(1);

      verify(predictor.assess(argThat(equals([
        [230, 330, 730],
        [530, 130, 930],
      ])), argThat(equals([[700], [900]])), metric)).called(1);
    });
  });
}
