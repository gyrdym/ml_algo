import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/softmax_regressor.dart';
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
      this.classNames,
      this.linkFunction, {
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        Matrix initialCoefficients,
        num positiveLabel = 1,
        num negativeLabel = 0,
        this.dtype = DType.float32,
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        _positiveLabel = positiveLabel,
        _negativeLabel = negativeLabel,
        coefficientsByClasses = optimizer.findExtrema(
          initialCoefficients: initialCoefficients,
          isMinimizingObjective: false,
        );

  @override
  final List<String> classNames;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  @override
  final Matrix coefficientsByClasses;

  final DType dtype;

  @override
  final LinkFunction linkFunction;

  final num _positiveLabel;

  final num _negativeLabel;

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

      final allZeroes = List.filled(
        coefficientsByClasses.columnsNum,
        _negativeLabel,
      );

      allZeroes[labelIdx] = _positiveLabel;

      return Vector.fromList(allZeroes, dtype: dtype);
    });

    return DataFrame.fromMatrix(
      classes,
      header: classNames,
    );
  }
}
