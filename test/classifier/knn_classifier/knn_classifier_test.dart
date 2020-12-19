import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_helpers/create_knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifier', () {
    final classLabelPrefix = 'class label';
    final data = DataFrame.fromSeries(
        [
          Series('first' , <num>[1, 1, 1, 1]),
          Series('second', <num>[2, 2, 2, 2]),
          Series('third' , <num>[2, 2, 2, 2]),
          Series('fourth', <num>[4, 4, 4, 4]),
          Series('fifth' , <num>[1, 3, 2, 1], isDiscrete: true),
        ],
    );

    final targetName = 'fifth';

    Kernel kernelMock;
    KernelFactory kernelFactoryMock;

    KnnSolver solverMock;
    KnnSolverFactory solverFactoryMock;

    KnnClassifier knnClassifierMock;
    KnnClassifierFactory knnClassifierFactoryMock;

    setUp(() {
      kernelMock = KernelMock();
      kernelFactoryMock = createKernelFactoryMock(kernelMock);

      solverMock = KnnSolverMock();
      solverFactoryMock = createKnnSolverFactoryMock(solverMock);

      knnClassifierMock = KnnClassifierMock();
      knnClassifierFactoryMock = createKnnClassifierFactoryMock(
          knnClassifierMock);

      knnClassifierInjector
        ..registerSingleton<KernelFactory>(() => kernelFactoryMock)
        ..registerSingleton<KnnSolverFactory>(() => solverFactoryMock)
        ..registerSingleton<KnnClassifierFactory>(() => knnClassifierFactoryMock);
    });

    tearDown(() {
      reset(kernelMock);
      reset(kernelFactoryMock);

      reset(solverMock);
      reset(solverFactoryMock);

      reset(knnClassifierMock);
      reset(knnClassifierFactoryMock);

      injector.clearAll();
      knnClassifierInjector.clearAll();
    });

    test('should call kernel factory with proper kernel type', () {
      createKnnClassifier(
        data,
        targetName,
        2,
        KernelType.uniform,
        Distance.cosine,
        classLabelPrefix,
        DType.float32,
      );

      verify(kernelFactoryMock.createByType(KernelType.uniform)).called(1);
    });

    test('should call solver factory with proper train features, train labels, '
        'k parameter, distance type and standardization flag', () {
      createKnnClassifier(
        data,
        targetName,
        2,
        KernelType.uniform,
        Distance.hamming,
        classLabelPrefix,
        DType.float32,
      );

      verify(solverFactoryMock.create(
        argThat(equals([
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
        ])),
        argThat(equals([
          [1],
          [3],
          [2],
          [1],
        ])),
        2,
        Distance.hamming,
        true,
      )).called(1);
    });

    test('should call KnnClassifierFactory in order to create a classifier', () {
      final classifier = createKnnClassifier(
        data,
        targetName,
        2,
        KernelType.uniform,
        Distance.cosine,
        classLabelPrefix,
        DType.float32,
      );

      verify(knnClassifierFactoryMock.create(
        targetName,
        [1, 3, 2],
        kernelMock,
        solverMock,
        classLabelPrefix,
        DType.float32,
      )).called(1);

      expect(classifier, same(knnClassifierMock));
    });

    test('should extract class label list from target column even if the '
        'latter is not marked as discrete', () {
      final data = DataFrame.fromSeries(
          [
            Series('first' , <num>[1, 1, 1, 1, 1, 1, 1, 1]),
            Series('target' , <num>[1, 3, 2, 1, 3, 3, 2, 1]),
          ],
      );

      createKnnClassifier(
        data,
        'target',
        2,
        KernelType.uniform,
        Distance.hamming,
        classLabelPrefix,
        DType.float32,
      );

      final expectedLabels = [1, 3, 2];

      verify(knnClassifierFactoryMock.create(
        any,
        expectedLabels,
        any,
        any,
        classLabelPrefix,
        any,
      )).called(1);
    });

    test('should throw an exception if target column does not exist in the '
        'train data', () {
      final actual = () => createKnnClassifier(
        data,
        'unknown_column',
        2,
        KernelType.uniform,
        Distance.hamming,
        classLabelPrefix,
        DType.float32,
      );

      expect(actual, throwsException);
    });
  });
}
