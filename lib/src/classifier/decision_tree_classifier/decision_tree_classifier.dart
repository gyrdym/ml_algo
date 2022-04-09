import 'dart:io';

import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_init_module.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs decision tree-based classification
///
/// Decision tree is an algorithm that recursively splits the input data into
/// subsets until the subsets conforming certain stop criteria are met.
///
/// Process of forming such a recursive subsets structure is called
/// decision tree learning. Once a decision tree learned, it may be used to
/// classify new samples with the same features that were used to learn the
/// tree.
abstract class DecisionTreeClassifier
    implements
        Assessable,
        Serializable,
        Retrainable<DecisionTreeClassifier>,
        Classifier {
  /// Parameters:
  ///
  /// [trainData] A [DataFrame] with observations that will be used by the
  /// classifier to learn a decision tree. Must contain [targetName] column.
  ///
  /// [targetName] A name of a column in [trainData] that contains class
  /// labels
  ///
  /// [minError] A value within the range 0..1 (both inclusive). The value
  /// denotes a minimal error on a single decision tree node and is used as a
  /// stop criteria to avoid farther decision's tree node splitting: if the
  /// node is good enough, there is no need to split it and thus it will become
  /// a leaf.
  ///
  /// [minSamplesCount] A minimal number of samples (observations) on the
  /// decision's tree node. The value is used as a stop criteria to avoid
  /// farther decision's tree node splitting: if the node contains less than or
  /// equal to [minSamplesCount] observations, the node turns into the leaf.
  ///
  /// [maxDepth] A maximum number of decision tree levels.
  ///
  /// [assessorType] Defines an assessment type that will be applied to a subset
  /// of data in order to decide how to split the subset while building the tree.
  /// Default value is [TreeAssessorType.gini]
  ///
  /// Possible values of [assessorType]
  ///
  /// [TreeAssessorType.gini] The algorithm makes a decision on how to split a
  /// subset of data based on the [Gini index](https://en.wikipedia.org/wiki/Gini_coefficient)
  ///
  /// [TreeAssessorType.majority] The algorithm makes a decision on how to split a
  /// subset of data based on a major class.
  factory DecisionTreeClassifier(
    DataFrame trainData,
    String targetName, {
    num minError = 0.5,
    int minSamplesCount = 1,
    int maxDepth = 10,
    DType dtype = dTypeDefaultValue,
    TreeAssessorType assessorType = TreeAssessorType.gini,
  }) =>
      initDecisionTreeModule().get<DecisionTreeClassifierFactory>().create(
            trainData,
            targetName,
            dtype,
            minError,
            minSamplesCount,
            maxDepth,
            assessorType,
          );

  /// Restores previously fitted classifier instance from the given [json]
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
  /// final classifier = DecisionTreeClassifier(
  ///   samples,
  ///   targetName,
  ///   minError: 0.3,
  ///   minSamplesCount: 1,
  ///   maxDepth: 3,
  /// );
  ///
  /// final pathToFile = './classifier.json';
  ///
  /// await classifier.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredClassifier = DecisionTreeClassifier.fromJson(json);
  ///
  /// // here you can use previously fitted restored classifier to make
  /// // some prediction, e.g. via `DecisionTreeClassifier.predict(...)`;
  /// ````
  factory DecisionTreeClassifier.fromJson(String json) =>
      initDecisionTreeModule()
          .get<DecisionTreeClassifierFactory>()
          .fromJson(json);

  /// A minimal error on a single decision tree node. It is used as a
  /// stop criteria to avoid farther decision's tree node splitting: if the
  /// node is good enough, there is no need to split it and thus it can be
  /// considered a leaf.
  ///
  /// The value is within the range 0..1 (both inclusive).
  ///
  /// The value is read-only, it's a hyperparameter of the model
  num get minError;

  /// A minimal number of samples (observations) on the
  /// decision's tree node. The value is used as a stop criteria to avoid
  /// farther decision's tree node splitting: if the node contains less than or
  /// equal to [minSamplesCount] observations, the node is considered a leaf.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get minSamplesCount;

  /// A maximum number of decision tree levels.
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get maxDepth;

  /// Defines an assessment type that was applied to a subset of data in
  /// order to decide how to split the subset while building the tree
  TreeAssessorType get assessorType;

  /// Saves tree as SVG-image. Example:
  ///
  /// ```dart
  /// final samples = (await fromCsv('path/to/dataset.csv'));
  /// final classifier = DecisionTreeClassifier(
  ///   samples,
  ///   'target',
  ///   minError: 0.3,
  ///   minSamplesCount: 5,
  ///   maxDepth: 4,
  /// );
  //
  //  await classifier.saveAsSvg('tree.svg');
  /// ```
  ///
  /// The file 'tree.svg' now contains a graphical representation of the tree
  Future<File> saveAsSvg(String filePath);
}
