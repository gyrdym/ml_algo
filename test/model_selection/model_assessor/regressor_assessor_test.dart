import 'dart:math' as math;

import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/regressor_assessor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('RegressorAssessor', () {
    final generator = math.Random();
    final metricFactoryMock = MockMetricFactory();
    final metricMock = MockMetric();
    final featureTargetSplitterMock = MockFeatureTargetSplitter();
    final assessor = RegressorAssessor(
      metricFactoryMock,
      featureTargetSplitterMock.split,
    );
    final metricType = MetricType.mape;
    final predictorMock = MockPredictor();
    final featuresNames = ['feature_1', 'feature_2', 'feature_3'];
    final targetNames = ['target_1'];
    final samplesHeader = [...featuresNames, ...targetNames];
    final samples = DataFrame([
      <num>[     1,  33,   -199, 1, 0, 0],
      <num>[-90002, 232, 889.20, 1, 0, 0],
      <num>[-12004,  19,    111, 0, 1, 0],
    ], headerExists: false, header: samplesHeader);
    final featuresMock = DataFrame([
      <num>[     1,  33,   -199],
      <num>[-90002, 232, 889.20],
      <num>[-12004,  19,    111],
    ], headerExists: false, header: featuresNames);
    final targetMock = DataFrame([
      <num>[100],
      <num>[200],
      <num>[0],
    ], headerExists: false, header: targetNames);
    final predictionMock = DataFrame([
      <num>[1000],
      <num>[2000],
      <num>[3000],
    ], headerExists: false, header: targetNames);
    final dtype = DType.float64;

    setUp(() {
      when(
        predictorMock.dtype,
      ).thenReturn(dtype);

      when(
        predictorMock.targetNames,
      ).thenReturn(targetNames);

      when(
        predictorMock.dtype,
      ).thenReturn(DType.float64);

      when(
        featureTargetSplitterMock.split(
          argThat(anything),
          targetNames: anyNamed('targetNames'),
        ),
      ).thenReturn([featuresMock, targetMock]);

      when(
        predictorMock.predict(
          argThat(anything),
        ),
      ).thenReturn(predictionMock);

      when(
        metricFactoryMock.createByType(
          argThat(anything),
        ),
      ).thenReturn(metricMock);

      when(
        metricMock.getScore(
          any,
          any,
        ),
      ).thenReturn(1.0);
    });

    tearDown(() {
      reset(metricFactoryMock);
      reset(metricMock);
      reset(featureTargetSplitterMock);
      reset(predictorMock);
    });

    test('should throw an exception if improper metric type is provided', () {
      final metricTypes = [MetricType.accuracy, MetricType.precision];

      metricTypes.forEach((metricType) {
        final actual = () => assessor.assess(predictorMock, metricType, samples);

        expect(actual, throwsA(isA<InvalidMetricTypeException>()));
      });
    });

    test('should create metric entity', () {
      assessor.assess(predictorMock, metricType, samples);

      verify(metricFactoryMock.createByType(metricType)).called(1);
    });

    test('should predict labels', () {
      assessor.assess(predictorMock, metricType, samples);

      verify(predictorMock.predict(featuresMock)).called(1);
    });

    test('should return score', () {
      final score = generator.nextDouble();

      when(
        metricMock.getScore(
          argThat(anything),
          argThat(anything),
        ),
      ).thenReturn(score);

      final actual = assessor.assess(predictorMock, metricType, samples);

      expect(actual, equals(score));
    });
  });
}
