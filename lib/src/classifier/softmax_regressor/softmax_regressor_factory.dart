import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class SoftmaxRegressorFactory {
  SoftmaxRegressor create(
      Matrix coefficientsByClasses,
      List<String> classNames,
      LinkFunction linkFunction,
      bool fitIntercept,
      num interceptScale,
      num positiveLabel,
      num negativeLabel,
      List<num> costPerIteration,
      DType dtype,
  );
}
