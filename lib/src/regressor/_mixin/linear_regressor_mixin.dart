import 'package:ml_algo/src/common/predictor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_linalg/linalg.dart';

mixin LinearRegressorMixin implements LinearRegressor, Predictor {
  @override
  Matrix predict(Matrix dataFrame) {
    final prediction = addInterceptIf(
        fitIntercept, dataFrame, interceptScale) * coefficients;
    return prediction;
  }
}
