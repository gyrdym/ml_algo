import 'package:ml_algo/src/classifier/linear/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/linear/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements SoftmaxRegressor {

  SoftmaxRegressorImpl(
      LinearOptimizer optimizer,
      this.classLabels,
      this.linkFunction, {
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        Matrix initialWeights,
        this.dtype = ParameterDefaultValues.dtype,
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        coefficientsByClasses = optimizer.findExtrema(
          initialCoefficients: initialWeights,
          isMinimizingObjective: false,
        );

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Matrix classLabels;

  @override
  final Matrix coefficientsByClasses;

  final DType dtype;

  @override
  final LinkFunction linkFunction;

  @override
  Matrix predict(Matrix features) {
    final processedFeatures = addInterceptIf(fitIntercept, features,
        interceptScale);
    return getProbabilities(processedFeatures, coefficientsByClasses,
        linkFunction)
        .mapRows((probabilities) {
          final labelIdx = probabilities.toList().indexOf(probabilities.max());
          return classLabels.getRow(labelIdx);
        });
  }
}
