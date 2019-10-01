import 'package:ml_algo/src/classifier/linear/linear_classifier.dart';
import 'package:ml_algo/src/classifier/linear/log_likelihood_optimizer_factory.dart';
import 'package:ml_algo/src/classifier/linear/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

/// A factory that creates different presets of logistic regressors
///
/// Logistic regression is an algorithm that solves a binary classification
/// problem. The algorithm uses maximization of the passed data likelihood.
/// In other words, the regressor iteratively tries to select coefficients,
/// that makes combination of passed features and these coefficients most
/// likely.
abstract class LogisticRegressor implements LinearClassifier, Assessable {
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// classifier to learn coefficients of the hyperplane, which divides the
  /// features space, forming classes of the features. Should contain target
  /// column id (index or name)
  ///
  /// [targetName] A string, that serves as a name of the target column (a
  /// column, that contains class labels or outcomes for the associated
  /// features)
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the solver. Default value is 100
  ///
  /// [initialLearningRate] A value, defining velocity of the convergence of the
  /// gradient descent solver. Default value is 1e-3
  ///
  /// [minCoefficientsUpdate] A minimum distance between weights vectors in two
  /// subsequent iterations. Uses as a condition of convergence in the
  /// solver. In other words, if difference is small, there is no reason to
  /// continue fitting. Default value is 1e-12
  ///
  /// [probabilityThreshold] A probability, on the basis of which it is decided,
  /// whether an observation relates to positive class label (label = 1) or
  /// negative class label (label = 0). The greater the probability, the more
  /// strict the classifier is. Default value is `0.5`
  ///
  /// [lambda] A coefficient of regularization. In gradient version of
  /// logistic regression L2 regularization is used
  ///
  /// [randomSeed] A seed, that will be passed to a random value generator,
  /// used by stochastic optimizers. Will be ignored, if the solver is not
  /// a stochastic. Remember, each time you run the regressor based on, for
  /// instance, stochastic gradient descent, with the same parameters, you will
  /// receive a different result. To avoid it, define [randomSeed]
  ///
  /// [batchSize] A size of data (in rows), that will be used for fitting per
  /// one iteration. If [batchSize] == `1` when stochastic gradient descent is
  /// used; if `1` < [batchSize] < `total number of rows`, when mini-batch
  /// gradient descent is used; if [batchSize] == `total number of rows`,
  /// when full-batch gradient descent is used
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`.
  ///
  /// [interceptScale] A value, defining a size of the intercept term
  ///
  /// [learningRateType] A value, defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32]
  factory LogisticRegressor(DataFrame fittingData, String targetName, {
    LinearOptimizerType optimizerType = LinearOptimizerType.vanillaGD,
    int iterationsLimit = 100,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double probabilityThreshold = 0.5,
    double lambda = 0.0,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    bool isFittingDataNormalized = false,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialCoefficientsType = InitialWeightsType.zeroes,
    Matrix initialCoefficients,
    DType dtype = DType.float32,
  }) {
    final dependencies = getDependencies();

    final linkFunctionFactory = dependencies
        .getDependency<LinkFunctionFactory>();

    final linkFunction = linkFunctionFactory
        .createByType(LinkFunctionType.inverseLogit, dtype: dtype);

    final splits = featuresTargetSplit(fittingData,
      targetNames: [targetName],
    ).toList();

    final points = splits[0].toMatrix();
    final labels = splits[1].toMatrix();

    final optimizer = createLogLikelihoodOptimizer(
      points,
      labels,
      LinkFunctionType.inverseLogit,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      randomSeed: randomSeed,
      batchSize: batchSize,
      learningRateType: learningRateType,
      initialWeightsType: initialCoefficientsType,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      isFittingDataNormalized: isFittingDataNormalized,
      dtype: dtype,
    );

    return LogisticRegressorImpl(
      optimizer,
      labels.uniqueRows(),
      linkFunction,
      probabilityThreshold: probabilityThreshold,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      initialWeights: initialCoefficients,
      dtype: dtype,
    );
  }
}
