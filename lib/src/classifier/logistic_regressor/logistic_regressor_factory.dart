import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LogisticRegressorFactory {
  LogisticRegressor create(
      String targetName,
      LinkFunction linkFunction,
      double probabilityThreshold,
      bool fitIntercept,
      double interceptScale,
      Matrix coefficientsByClasses,
      num negativeLabel,
      num positiveLabel,
      DType dtype,
  );
}
