import 'package:ml_algo/src/classifier/_helpers/log_likelihood_optimizer_factory.dart';
import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/logistic_regressor_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

/// A class, performing logistic regression-based classification.
///
/// Logistic regression is an algorithm that solves a binary classification
/// problem. The algorithm uses maximization of the passed data likelihood.
/// In other words, the regressor iteratively tries to select coefficients,
/// that makes combination of passed features and their coefficients most
/// likely.
abstract class LogisticRegressor implements LinearClassifier, Assessable {
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// classifier to learn coefficients of the hyperplane, which divides the
  /// features space, forming clusters of positive and negative classes of the
  /// observations. Must contain [targetName] column.
  ///
  /// [targetName] A string, that serves as a name of the target column (a
  /// column, that contains class labels or outcomes for the associated
  /// features).
  ///
  /// [optimizerType] Defines an algorithm of optimization, that will be used
  /// to find the best coefficients of log-likelihood cost function. Also
  /// defines, which regularization type (L1 or L2) one may use to learn a
  /// logistic regressor.
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the optimization algorithm. Default value is `100`.
  ///
  /// [initialLearningRate] A value, defining velocity of the convergence of the
  /// gradient descent optimizer. Default value is `1e-3`.
  ///
  /// [minCoefficientsUpdate] A minimum distance between coefficient vectors in
  /// two contiguous iterations. Uses as a condition of convergence in the
  /// optimization algorithm. If difference between the two vectors is small
  /// enough, there is no reason to continue fitting. Default value is `1e-12`
  ///
  /// [probabilityThreshold] A probability, on the basis of which it is decided,
  /// whether an observation relates to positive class label (see
  /// [positiveLabel] parameter) or to negative class label (see [negativeLabel]
  /// parameter). The greater the probability, the more strict the classifier
  /// is. Default value is `0.5`.
  ///
  /// [lambda] A coefficient of regularization. Uses to prevent the classifier's
  /// overfitting. The more the value of [lambda], the more regular the
  /// coefficients of log-likelihood cost function are. Extremely large [lambda]
  /// may decrease the coefficients to nothing, otherwise too small [lambda] may
  /// be a cause of too large absolute values of the coefficients.
  ///
  /// [randomSeed] A seed, that will be passed to a random value generator,
  /// used by stochastic optimizers. Will be ignored, if the solver is not
  /// a stochastic. Remember, each time you run the regressor based on, for
  /// instance, stochastic gradient descent, with the same parameters, you will
  /// receive a different result. To avoid it, define [randomSeed]
  ///
  /// [batchSize] A size of data (in rows), that will be used for fitting per
  /// one iteration. Applicable not for all optimizers. If gradient-based
  /// optimizer uses and If [batchSize] == `1`, stochastic mode will be
  /// activated; if `1` < [batchSize] < `total number of rows`, mini-batch mode
  /// will be activated; if [batchSize] == `total number of rows`, full-batch
  /// mode will be activated.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`. Intercept in 2-dimensional space is a bias of the line (relative
  /// to X-axis) to be learned by the classifier
  ///
  /// [interceptScale] A value, defining a size of the intercept.
  ///
  /// [isFittingDataNormalized] Defines, whether the [fittingData] normalized
  /// or not. Normalization should be performed column-wise. Normalized data
  /// may be needed for some optimizers (e.g., for
  /// [LinearOptimizerType.vanillaCD])
  ///
  /// [learningRateType] A value, defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [initialCoefficientsType] Defines the coefficients, that will be
  /// autogenerated before the first iteration of optimization. By default,
  /// all the autogenerated coefficients are equal to zeroes at the start.
  /// If [initialCoefficients] are provided, the parameter will be ignored
  ///
  /// [initialCoefficients] Coefficients to be used in the first iteration of
  /// optimization algorithm. [initialCoefficients] is a vector, length of which
  /// must be equal to the number of features in [fittingData] : in case of
  /// logistic regression only one column from [fittingData] is used as a
  /// prediction target column, thus the number of features is equal to
  /// the number of columns in [fittingData] minus 1 (target column).
  ///
  /// [positiveLabel] Defines the value, that will be used for `positive` class.
  /// By default, `1`.
  ///
  /// [negativeLabel] Defines the value, that will be used for `negative` class.
  /// By default, `0`.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32]
  factory LogisticRegressor(
      DataFrame fittingData,
      String targetName, {
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
    InitialCoefficientsType initialCoefficientsType =
        InitialCoefficientsType.zeroes,
    Vector initialCoefficients,
    num positiveLabel = 1,
    num negativeLabel = 0,
    DType dtype = DType.float32,
  }) {
    final dependencies = getDependencies();

    final linkFunctionFactory = dependencies
        .getDependency<LinkFunctionFactory>();

    final linkFunction = linkFunctionFactory
        .createByType(LinkFunctionType.inverseLogit, dtype: dtype);

    final optimizer = createLogLikelihoodOptimizer(
      fittingData,
      [targetName],
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
      targetName,
      linkFunction,
      probabilityThreshold: probabilityThreshold,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      initialCoefficients: initialCoefficients,
      positiveLabel: positiveLabel,
      negativeLabel: negativeLabel,
      dtype: dtype,
    );
  }
}
