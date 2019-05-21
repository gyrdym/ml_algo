import 'dart:typed_data';

import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_impl.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
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
      final allObservations = Matrix.fromList([
        [330, 930, 130],
        [630, 830, 230],
        [730, 730, 330],
        [830, 630, 430],
        [930, 530, 530],
        [130, 430, 630],
        [230, 330, 730],
        [430, 230, 830],
        [530, 130, 930],
      ]);
      final allOutcomes = Matrix.fromList([
        [100],[200],[300],[400],[500],[600],[700],[800],[900],
      ]);
      final metric = MetricType.mape;
      final splitter = createSplitter([[0,2,4],[6, 8]]);
      final predictor = PredictorMock();
      final validator = CrossValidatorImpl(DType.float32, splitter);

      var score = 20.0;
      when(predictor.assess(any, any, any))
          .thenAnswer((Invocation inv) => score = score + 10);

      final actual = validator.evaluate((observations, outcomes) => predictor,
          allObservations, allOutcomes, metric);

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

    test('should throw an exception if observations number and outcomes number '
        'mismatch', () {
      final allObservations = Matrix.fromList([
        [330, 930, 130],
        [630, 830, 230],
      ]);
      final allOutcomes = Matrix.fromList([
        [100],
      ]);
      final metric = MetricType.mape;
      final splitter = SplitterMock();
      final predictor = PredictorMock();
      final validator = CrossValidatorImpl(DType.float32, splitter);

      expect(() => validator.evaluate((observations, outcomes) => predictor,
          allObservations, allOutcomes, metric), throwsException);
    });
  });
}
