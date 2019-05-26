import 'package:ml_algo/src/classifier/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/logit/inverse_logit_link_function.dart';
import 'package:ml_algo/src/optimizer/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/linear/linear_optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/linear/linear_optimizer_factory_impl.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class GradientLogisticRegressor with LinearClassifierMixin
    implements LogisticRegressor {
  GradientLogisticRegressor(
      Matrix trainingFeatures,
      Matrix trainingOutcomes, {
        // public arguments
        int iterationsLimit = DefaultParameterValues.iterationsLimit,
        double initialLearningRate = DefaultParameterValues.initialLearningRate,
        double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
        double lambda,
        int randomSeed,
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        LearningRateType learningRateType = LearningRateType.constant,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
        Matrix initialWeights,
        this.dtype = DefaultParameterValues.dtype,
        this.probabilityThreshold = 0.5,

        LinearOptimizerFactory optimizerFactory =
          const LinearOptimizerFactoryImpl(),
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        linkFunction = InverseLogitLinkFunction(dtype),
        classLabels = trainingOutcomes.uniqueRows(),
        coefficientsByClasses = optimizerFactory.gradient(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          dtype: dtype,
          costFunction: LogLikelihoodCost(InverseLogitLinkFunction(dtype)),
          learningRateType: learningRateType,
          initialWeightsType: initialWeightsType,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minWeightsUpdate,
          iterationLimit: iterationsLimit,
          lambda: lambda,
          batchSize: batchSize,
          randomSeed: randomSeed,
        ).findExtrema(
            initialWeights: initialWeights,
            isMinimizingObjective: false,
        );

  @override
  final Matrix coefficientsByClasses;

  @override
  final Matrix classLabels;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  final DType dtype;

  final double probabilityThreshold;

  @override
  final LinkFunction linkFunction;

  @override
  Matrix predictClasses(Matrix features) {
    final processedFeatures = addInterceptIf(fitIntercept, features,
        interceptScale);
    final classesSource = getProbabilities(processedFeatures,
        coefficientsByClasses, linkFunction)
        .getColumn(0)
        // TODO: use SIMD
        .map((value) => value >= probabilityThreshold ? 1.0 : 0.0)
        .toList(growable: false);
    return Matrix.fromColumns([Vector.fromList(classesSource)]);
  }
}
