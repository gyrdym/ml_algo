import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/_mixin/linear_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LinearRegressorImpl with AssessablePredictorMixin, LinearRegressorMixin
    implements LinearRegressor {

  LinearRegressorImpl(this.coefficients, {
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Matrix initialCoefficients,
    this.dtype = DType.float32,
  }) :
    fitIntercept = fitIntercept,
    interceptScale = interceptScale;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Vector coefficients;

  final DType dtype;
}
