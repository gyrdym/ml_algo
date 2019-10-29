import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('SoftmaxRegressorFactoryImpl', () {
    test('should create a SoftmaxRegressorImpl instance', () {
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
      final dtype = DType.float32;

      final regressor = factory.create(
        coefficientsByClasses,
        classNames,
        linkFunction,
        fitIntercept,
        interceptScale,
        positiveLabel,
        negativeLabel,
        dtype,
      );

      expect(regressor, isA<SoftmaxRegressorImpl>());
    });
  });
}
