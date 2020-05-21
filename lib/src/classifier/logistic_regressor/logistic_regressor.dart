import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_helpers/create_logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_helpers/create_logistic_regressor_from_json.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

/// Logistic regression-based classification.
///
/// Logistic regression is an algorithm that solves a binary classification
/// problem. The algorithm uses maximization of the passed data likelihood.
/// In other words, the regressor iteratively tries to select coefficients
/// that makes combination of passed features and their coefficients most
/// likely.
abstract class LogisticRegressor implements
    LinearClassifier, Assessable, Serializable {

  /// Parameters:
  ///
  /// [trainData] A [DataFrame] with observations, that will be used by the
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
  /// [lambda] A coefficient of regularization. Uses to prevent the regressor's
  /// overfitting. The more the value of [lambda], the more regular the
  /// coefficients of the equation of the predicting hyperplane are. Extremely
  /// large [lambda] may decrease the coefficients to nothing, otherwise too
  /// small [lambda] may be a cause of too large absolute values of the
  /// coefficients, that is also bad.
  ///
  /// [regularizationType] A way the coefficients of the classifier will be
  /// regularized to prevent a model overfitting.
  ///
  /// [randomSeed] A seed, that will be passed to a random value generator,
  /// used by stochastic optimizers. Will be ignored, if the solver cannot be
  /// stochastic. Remember, each time you run the stochastic regressor with the
  /// same parameters but with unspecified [randomSeed], you will receive
  /// different results. To avoid it, define [randomSeed]
  ///
  /// [batchSize] A size of data (in rows), that will be used for fitting per
  /// one iteration. Applicable not to all optimizers. If gradient-based
  /// optimizer uses and If [batchSize] == `1`, stochastic mode will be
  /// activated; if `1` < [batchSize] < `total number of rows`, mini-batch mode
  /// will be activated; if [batchSize] == `total number of rows`, full-batch
  /// mode will be activated.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`. Intercept in 2-dimensional space is a bias of the line (relative
  /// to X-axis).
  ///
  /// [interceptScale] A value, defining a size of the intercept.
  ///
  /// [isFittingDataNormalized] Defines, whether the [trainData] normalized
  /// or not. Normalization should be performed column-wise. Normalized data
  /// may be needed for some optimizers (e.g., for
  /// [LinearOptimizerType.coordinate])
  ///
  /// [learningRateType] A value, defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [initialCoefficientsType] Defines the coefficients, that will be
  /// autogenerated at the first optimization iteration. By default,
  /// all the autogenerated coefficients are equal to zeroes at the start.
  /// If [initialCoefficients] are provided, the parameter will be ignored
  ///
  /// [initialCoefficients] Coefficients to be used in the first iteration of
  /// optimization algorithm. [initialCoefficients] is a vector, length of which
  /// must be equal to the number of features in [trainData] : in case of
  /// logistic regression only one column from [trainData] is used as a
  /// prediction target column, thus the number of features is equal to
  /// the number of columns in [trainData] minus 1 (target column). Keep in
  /// mind, that if your model considers intercept term, [initialCoefficients]
  /// should contain an extra element in the beginning of the vector and it
  /// denotes the intercept term coefficient
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
      DataFrame trainData,
      String targetName, {
    LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
    int iterationsLimit = 100,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double probabilityThreshold = 0.5,
    double lambda = 0.0,
    RegularizationType regularizationType,
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
  }) => createLogisticRegressor(
    trainData,
    targetName,
    optimizerType,
    iterationsLimit,
    initialLearningRate,
    minCoefficientsUpdate,
    probabilityThreshold,
    lambda,
    regularizationType,
    randomSeed,
    batchSize,
    fitIntercept,
    interceptScale,
    isFittingDataNormalized,
    learningRateType,
    initialCoefficientsType,
    initialCoefficients ?? Vector.empty(dtype: dtype),
    positiveLabel,
    negativeLabel,
    dtype,
  );

  /// Restores previously fitted classifier instance from the [json]
  ///
  /// ````dart
  /// import 'dart:io';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// final data = <Iterable>[
  ///   ['feature 1', 'feature 2', 'feature 3', 'outcome']
  ///   [        5.0,         7.0,         6.0,       1.0],
  ///   [        1.0,         2.0,         3.0,       0.0],
  ///   [       10.0,        12.0,        31.0,       0.0],
  ///   [        9.0,         8.0,         5.0,       0.0],
  ///   [        4.0,         0.0,         1.0,       1.0],
  /// ];
  /// final targetName = 'outcome';
  /// final samples = DataFrame(data, headerExists: true);
  /// final classifier = LogisticRegressor(
  ///   samples,
  ///   targetName,
  ///   iterationsLimit: 2,
  ///   learningRateType: LearningRateType.constant,
  ///   initialLearningRate: 1.0,
  ///   batchSize: 5,
  ///   fitIntercept: true,
  ///   interceptScale: 3.0,
  /// );
  ///
  /// final pathToFile = './classifier.json';
  ///
  /// await classifier.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredClassifier = LogisticRegressor.fromJson(json);
  ///
  /// // here you can use previously fitted restored classifier to make
  /// // some prediction, e.g. via `restoredClassifier.predict(...)`;
  /// ````
  factory LogisticRegressor.fromJson(String json) =>
      createLogisticRegressorFromJson(json);
}
