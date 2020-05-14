import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class LogisticRegressorFactoryImpl implements LogisticRegressorFactory {
  const LogisticRegressorFactoryImpl();

  @override
  LogisticRegressor create(
      String targetName,
      LinkFunction linkFunction,
      num probabilityThreshold,
      bool fitIntercept,
      num interceptScale,
      Matrix coefficientsByClasses,
      num negativeLabel,
      num positiveLabel,
      DType dtype,
  ) => LogisticRegressorImpl(
    [targetName],
    linkFunction,
    fitIntercept,
    interceptScale,
    coefficientsByClasses,
    probabilityThreshold,
    negativeLabel,
    positiveLabel,
    dtype,
  );
}
