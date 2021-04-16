import 'package:ml_algo/src/common/exception/outdated_json_schema_exception.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../helpers.dart';
import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('KnnRegressorImpl', () {
    final targetName = 'target';
    final kernelType = KernelType.epanechnikov;
    final distanceType = Distance.manhattan;
    final k = 3209;
    final solver = MockKnnSolver();
    final kernel = MockKernel();
    final dtype = DType.float32;
    final retrainedModelMock = MockKnnRegressor();
    final regressorFactory = createKnnRegressorFactoryMock(retrainedModelMock);
    final regressor = KnnRegressorImpl(
      targetName,
      solver,
      kernel,
      dtype,
    );

    tearDown(() {
      injector.clearAll();
      knnRegressorInjector.clearAll();
    });

    group('predict', () {
      tearDown(() {
        reset(solver);
        reset(kernel);
      });

      test('should throw an exception if no features are provided', () {
        final data = DataFrame.fromMatrix(Matrix.empty());

        expect(() => regressor.predict(data), throwsException);
      });

      test('should return weighted average of outcomes of found k neighbours as '
          'a prediction for each test feature record', () {
        final testFeatureMatrix = Matrix.fromList([
          [10, 20, 30, 50,  10],
          [22, 20, 27, 50,  34],
          [44, 20, 92, 54,  51],
          [44, 18, 09, 50,  00],
          [10, 20, 30, 51,  00],
          [43, 95, 10, 33, -60],
        ]);

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        final foundNeighbours = [
          [
            Neighbour(10, Vector.fromList([-10])),
            Neighbour(20, Vector.fromList([2])),
            Neighbour(30, Vector.fromList([44])),
          ],
          [
            Neighbour(10, Vector.fromList([93])),
            Neighbour(100, Vector.fromList([1])),
            Neighbour(1000, Vector.fromList([-10])),
          ],
          [
            Neighbour(-10, Vector.fromList([1000])),
            Neighbour(5, Vector.fromList([100])),
            Neighbour(12, Vector.fromList([10])),
          ],
          [
            Neighbour(0, Vector.fromList([263])),
            Neighbour(0, Vector.fromList([921])),
            Neighbour(13, Vector.fromList([122])),
          ],
          [
            Neighbour(1000, Vector.fromList([1])),
            Neighbour(2000, Vector.fromList([1])),
            Neighbour(3000, Vector.fromList([1])),
          ],
          [
            Neighbour(1, Vector.fromList([0])),
            Neighbour(2, Vector.fromList([0])),
            Neighbour(3, Vector.fromList([0])),
          ],
        ];

        when(kernel.getWeightByDistance(10, any)).thenReturn(30);
        when(kernel.getWeightByDistance(20, any)).thenReturn(29);
        when(kernel.getWeightByDistance(30, any)).thenReturn(28);

        when(kernel.getWeightByDistance(100, any)).thenReturn(10);
        when(kernel.getWeightByDistance(1000, any)).thenReturn(0);

        when(kernel.getWeightByDistance(-10, any)).thenReturn(0);
        when(kernel.getWeightByDistance(5, any)).thenReturn(-1);
        when(kernel.getWeightByDistance(12, any)).thenReturn(-2);

        when(kernel.getWeightByDistance(0, any)).thenReturn(10);
        when(kernel.getWeightByDistance(13, any)).thenReturn(1);

        when(kernel.getWeightByDistance(2000, any)).thenReturn(4);
        when(kernel.getWeightByDistance(3000, any)).thenReturn(0);

        when(kernel.getWeightByDistance(1, any)).thenReturn(3);
        when(kernel.getWeightByDistance(2, any)).thenReturn(2);
        when(kernel.getWeightByDistance(3, any)).thenReturn(1);

        when(solver.findKNeighbours(testFeatureMatrix))
            .thenReturn(foundNeighbours);

        final actual = regressor.predict(testFeatures);

        final expected = [
          [ 11.379],
          [ 70.000],
          [ 40.000],
          [569.619],
          [  1.000],
          [  0.000],
        ];

        expect(actual.rows, iterable2dAlmostEqualTo(expected, 1e-3));
      });

      test('should return a dataframe with a proper header', () {
        when(solver.findKNeighbours(any)).thenReturn([
          [
            Neighbour(23, Vector.fromList([1]))
          ],
        ]);

        when(
          kernel.getWeightByDistance(
            any,
            any,
          )
        ).thenReturn(1);

        final data = DataFrame.fromMatrix(Matrix.fromList([
          [1, 2, 3],
        ]));

        expect(regressor.predict(data).header, equals([targetName]));
      });
    });

    group('retrain', () {
      final retrainingData = DataFrame([[100, -200, 300, 444.55]]);

      setUp(() {
        when(solver.distanceType).thenReturn(distanceType);
        when(solver.k).thenReturn(k);
        when(kernel.type).thenReturn(kernelType);

        knnRegressorInjector
            .registerSingleton<KnnRegressorFactory>(
                () => regressorFactory);
      });

      tearDown(() {
        reset(solver);
        reset(kernel);
      });

      test('should call a factory while retraining the model', () {
        regressor.retrain(retrainingData);

        verify(regressorFactory.create(
          retrainingData,
          targetName,
          k,
          kernelType,
          distanceType,
          dtype,
        )).called(1);
      });

      test('should return a new instance as a retrained model', () {
        final retrainedModel = regressor.retrain(retrainingData);

        expect(retrainedModel, same(retrainedModelMock));
        expect(retrainedModel, isNot(same(regressor)));
      });

      test('should throw exception if the model schema is outdated, '
          'schemaVersion is null', () {
        final model = KnnRegressorImpl(
          targetName,
          solver,
          kernel,
          dtype,
          schemaVersion: null,
        );

        expect(() => model.retrain(retrainingData),
            throwsA(isA<OutdatedJsonSchemaException>()));
      });
    });
  });
}
