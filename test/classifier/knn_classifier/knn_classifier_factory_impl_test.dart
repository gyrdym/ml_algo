import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierFactoryImpl', () {
    final kernelFactoryMock = KernelFunctionFactoryMock();
    final kernelMock = KernelMock();
    final solverFactoryMock = KnnSolverFactoryMock();
    final solverMock = KnnSolverMock();
    final factory = KnnClassifierFactoryImpl(
      kernelFactoryMock,
      solverFactoryMock,
    );
    final targetName = 'target';
    final header = ['col_1', 'col_2', targetName];
    final features = [
      [1, 2],
      [0, 4],
      [8, 7],
    ];
    final outcomes = [
      [3],
      [5],
      [6],
    ];
    final trainData = DataFrame([
      header,
      [...features[0], ...outcomes[0]],
      [...features[1], ...outcomes[1]],
      [...features[2], ...outcomes[2]],
    ]);
    final k = 10;
    final kernelType = KernelType.epanechnikov;
    final distanceType = Distance.hamming;
    final classLabelPrefix = 'label';
    final dtype = DType.float32;

    setUp(() {
      when(kernelFactoryMock.createByType(any as KernelType))
          .thenReturn(kernelMock);
      when(solverFactoryMock.create(
          any as Matrix,
          any as Matrix,
          any as int,
          any as Distance,
          any as bool)
      ).thenReturn(solverMock);
    });

    tearDown(() {
      reset(kernelFactoryMock);
      reset(kernelMock);
      reset(solverFactoryMock);
      reset(solverMock);
    });

    test('should create a KnnClassifierImpl instance', () {
      final actual = factory.create(
        trainData,
        targetName,
        k,
        kernelType,
        distanceType,
        classLabelPrefix,
        dtype,
      );

      expect(actual, isA<KnnClassifierImpl>());
    });

    test('should call kernel factory with proper kernel type', () {
      factory.create(
        trainData,
        targetName,
        k,
        kernelType,
        distanceType,
        classLabelPrefix,
        dtype,
      );

      verify(kernelFactoryMock.createByType(kernelType)).called(1);
    });

    test('should call solver factory with proper train features, train labels, '
        'k parameter, distance type and standardization flag', () {
      factory.create(
        trainData,
        targetName,
        k,
        kernelType,
        distanceType,
        classLabelPrefix,
        dtype,
      );

      verify(solverFactoryMock.create(
        argThat(equals(features)) as Matrix,
        argThat(equals(outcomes)) as Matrix,
        k,
        distanceType,
        true,
      )).called(1);
    });

    test('should extract class label list from target column even if the '
        'latter is not marked as discrete', () {
      final trainData = DataFrame.fromSeries(
        [
          Series('first' , <num>[1, 1, 1, 1, 1, 1, 1, 1]),
          Series(targetName , <num>[1, 3, 2, 1, 3, 3, 2, 1]),
        ],
      );

      final classifier = factory.create(
        trainData,
        targetName,
        k,
        kernelType,
        distanceType,
        classLabelPrefix,
        dtype,
      ) as KnnClassifierImpl;

      expect(classifier.classLabels, [1, 3, 2]);
    });

    test('should throw an exception if target column does not exist in the '
        'train data', () {
      final actual = () => factory.create(
        trainData,
        'some_unknown_column_name',
        k,
        kernelType,
        distanceType,
        classLabelPrefix,
        dtype,
      );

      expect(actual, throwsException);
    });
  });
}
