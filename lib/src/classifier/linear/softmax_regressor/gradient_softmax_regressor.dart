import 'package:ml_algo/src/classifier/_mixin/asessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/_mixin/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/linear/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/cost_function/log_likelihood.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer_factory.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer_factory_impl.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class GradientSoftmaxRegressor with LinearClassifierMixin,
    AssessableClassifierMixin implements SoftmaxRegressor {

  GradientSoftmaxRegressor(
      Matrix trainingFeatures,
      Matrix trainingOutcomes, {
        int iterationsLimit = ParameterDefaultValues.iterationsLimit,
        double initialLearningRate = ParameterDefaultValues.initialLearningRate,
        double minWeightsUpdate = ParameterDefaultValues.minCoefficientsUpdate,
        double lambda,
        int randomSeed,
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        Matrix initialWeights,
        LearningRateType learningRateType = LearningRateType.constant,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,

        this.dtype = ParameterDefaultValues.dtype,

        LinearOptimizerFactory optimizerFactory =
          const LinearOptimizerFactoryImpl(),
      }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        linkFunction = SoftmaxLinkFunction(dtype),
        classLabels = trainingOutcomes.uniqueRows(),
        coefficientsByClasses = optimizerFactory.gradient(
          addInterceptIf(fitIntercept, trainingFeatures, interceptScale),
          trainingOutcomes,
          dtype: dtype,
          costFunction: LogLikelihoodCost(SoftmaxLinkFunction(dtype)),
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
  Matrix predictClasses(Matrix features) {
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
