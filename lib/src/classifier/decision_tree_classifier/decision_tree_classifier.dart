import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_helper/create_decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_helper/create_decision_tree_classifier_from_json.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs decision tree-based classification
///
/// Decision tree is an algorithm that recursively splits the input data into
/// subsets until the subsets conforming certain stop criteria are found.
///
/// Process of forming such a recursive subsets structure is called
/// decision tree learning. Once a decision tree learned, it may be used to
/// classify new samples with the same features that were used to learn the
/// tree.
abstract class DecisionTreeClassifier implements
    Classifier, Assessable, Serializable {
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
  factory DecisionTreeClassifier(
      DataFrame trainData,
      String targetName, {
    num minError,
    int minSamplesCount,
    int maxDepth,
    DType dtype = DType.float32,
  }) => createDecisionTreeClassifier(
    trainData,
    targetName,
    minError,
    minSamplesCount,
    maxDepth,
    dtype,
  );

  factory DecisionTreeClassifier.fromJson(String json) =>
      createDecisionTreeClassifierFromJson(json);
}
