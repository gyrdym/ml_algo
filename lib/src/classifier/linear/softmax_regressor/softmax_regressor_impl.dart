import 'package:ml_algo/src/classifier/linear/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/linear/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements SoftmaxRegressor {

  SoftmaxRegressorImpl(
      LinearOptimizer optimizer,
      this._classNames,
      this.classLabels,
      this.linkFunction, {
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        Matrix initialWeights,
        this.dtype = DType.float32,
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        coefficientsByClasses = optimizer.findExtrema(
          initialCoefficients: initialWeights,
          isMinimizingObjective: false,
        );

  @override
  final Iterable<String> _classNames;

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
  DataFrame predict(DataFrame features) {
    final processedFeatures = addInterceptIf(
      fitIntercept,
      features.toMatrix(),
      interceptScale,
    );

    final classes = getProbabilities(
        processedFeatures,
        coefficientsByClasses,
        linkFunction
    ).mapRows((probabilities) {
      final labelIdx = probabilities
          .toList()
          .indexOf(probabilities.max());
      final allZeroes = List.filled(coefficientsByClasses.columnsNum, 0);

      allZeroes[labelIdx] = 1;

      return Vector.fromList(allZeroes, dtype: dtype);
    });

    return DataFrame.fromMatrix(
      classes,
      header: _classNames,
    );
  }
}
