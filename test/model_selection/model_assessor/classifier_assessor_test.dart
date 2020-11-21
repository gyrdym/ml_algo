import 'dart:math' as math;

import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('ClassifierAssessor', () {
    final generator = math.Random();
    final metricFactoryMock = MetricFactoryMock();
    final metricMock = MetricMock();
    final encoderFactoryMock = EncoderFactoryMock();
    final encoderMock = EncoderMock();
    final featureTargetSplitterMock = FeatureTargetSplitterMock();
    final classLabelsNormalizerMock = ClassLabelsNormalizerMock();
    final assessor = ClassifierAssessorImpl(
      metricFactoryMock,
      encoderFactoryMock,
      featureTargetSplitterMock,
      classLabelsNormalizerMock,
    );
    final metricType = MetricType.precision;
    final classifierMock = ClassifierMock();
    final featuresNames = ['feature_1', 'feature_2', 'feature_3'];
    final targetNames = ['target_1', 'target_2', 'target_2'];
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
      <num>[1, 0, 0],
      <num>[1, 0, 0],
      <num>[0, 1, 0],
    ], headerExists: false, header: targetNames);
    final predictionMock = DataFrame([
      <num>[0, 0, 1],
      <num>[0, 0, 1],
      <num>[1, 0, 0],
    ], headerExists: false, header: targetNames);
    final dtype = DType.float64;
    final positiveLabel = 100;
    final negativeLabel = -100;

    setUp(() {
      when(
          encoderFactoryMock.createOneHot(
            argThat(anything),
            featureIds: anyNamed('featureIds'),
            featureNames: anyNamed('featureNames'),
            headerPrefix: anyNamed('headerPrefix'),
            headerPostfix: anyNamed('headerPostfix'),
          )
      ).thenReturn(encoderMock);
      when(classifierMock.dtype).thenReturn(dtype);
      when(
          classifierMock.targetNames,
      ).thenReturn(targetNames);
      when(
        classifierMock.dtype,
      ).thenReturn(DType.float64);
      when(
          featureTargetSplitterMock.split(
            argThat(anything),
            targetNames: anyNamed('targetNames'),
          )
      ).thenReturn([featuresMock, targetMock]);
      when(
          classifierMock.predict(
            argThat(anything),
          ),
      ).thenReturn(predictionMock);
      when(
          encoderMock.process(
            argThat(anything),
          ),
      ).thenReturn(predictionMock);
      when(
          metricFactoryMock.createByType(
            argThat(anything),
          ),
      ).thenReturn(metricMock);
    });

    tearDown(() {
      reset(metricFactoryMock);
      reset(metricMock);
      reset(encoderFactoryMock);
      reset(encoderMock);
      reset(featureTargetSplitterMock);
      reset(classLabelsNormalizerMock);
    });

    test('should throw an exception if improper metric type is provided', () {
      final metricTypes = [MetricType.mape, MetricType.rmse];

      metricTypes.forEach((metricType) {
        final actual = () => assessor.assess(classifierMock, metricType, samples);

        expect(actual, throwsA(isA<InvalidMetricTypeException>()));
      });
    });

    test('should create metric entity', () {
      assessor.assess(classifierMock, metricType, samples);

      verify(metricFactoryMock.createByType(metricType)).called(1);
    });

    test('should encode predicted target column if it is not encoded', () {
      when(classifierMock.targetNames).thenReturn(['target']);

      assessor.assess(classifierMock, metricType, samples);

      verify(encoderMock.process(predictionMock)).called(1);
    });

    test('should encode original target column if it is not encoded', () {
      when(classifierMock.targetNames).thenReturn(['target']);

      assessor.assess(classifierMock, metricType, samples);

      verify(encoderMock.process(targetMock)).called(1);
    });

    test('should normalize predicted class labels if predefined labels for '
        'positive and negative classes exist', () {
      when(classifierMock.positiveLabel).thenReturn(positiveLabel);
      when(classifierMock.negativeLabel).thenReturn(negativeLabel);

      assessor.assess(classifierMock, metricType, samples);

      verify(
        classLabelsNormalizerMock.normalize(
          predictionMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      ).called(1);
    });

    test('should not normalize predicted class labels if predefined labels for '
        'positive and negative classes do not exist', () {
      when(classifierMock.positiveLabel).thenReturn(null);
      when(classifierMock.negativeLabel).thenReturn(null);

      assessor.assess(classifierMock, metricType, samples);

      verifyNever(
        classLabelsNormalizerMock.normalize(
          predictionMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      );
    });

    test('should not normalize predicted class labels if at least one '
        'predefined class label does not exist', () {
      when(classifierMock.positiveLabel).thenReturn(positiveLabel);
      when(classifierMock.negativeLabel).thenReturn(null);

      assessor.assess(classifierMock, metricType, samples);

      verifyNever(
        classLabelsNormalizerMock.normalize(
          predictionMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      );
    });

    test('should normalize original class labels if predefined labels for '
        'positive and negative classes exist', () {
      when(classifierMock.positiveLabel).thenReturn(positiveLabel);
      when(classifierMock.negativeLabel).thenReturn(negativeLabel);

      assessor.assess(classifierMock, metricType, samples);

      verify(
        classLabelsNormalizerMock.normalize(
          targetMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      ).called(1);
    });

    test('should not normalize original class labels if predefined labels for '
        'positive and negative classes do not exist', () {
      when(classifierMock.positiveLabel).thenReturn(null);
      when(classifierMock.negativeLabel).thenReturn(null);

      assessor.assess(classifierMock, metricType, samples);

      verifyNever(
        classLabelsNormalizerMock.normalize(
          targetMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      );
    });

    test('should not normalize original class labels if at least one '
        'predefined class label does not exist', () {
      when(classifierMock.positiveLabel).thenReturn(null);
      when(classifierMock.negativeLabel).thenReturn(negativeLabel);

      assessor.assess(classifierMock, metricType, samples);

      verifyNever(
        classLabelsNormalizerMock.normalize(
          targetMock.toMatrix(dtype), positiveLabel, negativeLabel,
        ),
      );
    });

    test('should return score', () {
      final score = generator.nextDouble();

      when(
          metricMock.getScore(argThat(anything), argThat(anything)),
      ).thenReturn(score);

      final actual = assessor.assess(classifierMock, metricType, samples);

      expect(actual, equals(score));
    });
  });
}
