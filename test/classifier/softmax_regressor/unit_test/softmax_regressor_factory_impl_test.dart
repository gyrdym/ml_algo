import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../../../mocks.dart';

void main() {
  group('SoftmaxRegressorFactoryImpl', () {
    final factory = const SoftmaxRegressorFactoryImpl();
    final coefficientsByClasses = Matrix.fromList([
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3],
    ]);
    final classNames = ['class 1', 'class 2', 'class 3'];
    final linkFunction = LinkFunctionMock();
    final fitIntercept = false;
    final interceptScale = 1;
    final positiveLabel = 1;
    final negativeLabel = -1;
    final costPerIteration = [1, 2, 3];
    final dtype = DType.float32;
    SoftmaxRegressor regressor;

    setUp(() {
      regressor = factory.create(
        coefficientsByClasses,
        classNames,
        linkFunction,
        fitIntercept,
        interceptScale,
        positiveLabel,
        negativeLabel,
        costPerIteration,
        dtype,
      );
    });

    test('should create a SoftmaxRegressorImpl instance', () {
      expect(regressor, isA<SoftmaxRegressorImpl>());
    });

    test('should persist data passed to the `create` method', () {
      expect(regressor.costPerIteration, costPerIteration);
      expect(regressor.dtype, dtype);
      expect(regressor.classNames, classNames);
      expect(regressor.interceptScale, interceptScale);
      expect(regressor.linkFunction, linkFunction);
      expect(regressor.fitIntercept, fitIntercept);
    });
  });
}
