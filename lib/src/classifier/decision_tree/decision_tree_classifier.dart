import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree/greedy_classifier_dependencies.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

abstract class DecisionTreeClassifier implements Classifier, Assessable {
  factory DecisionTreeClassifier.greedy(
      DataSet data,
      double minError,
      int minSamplesCount,
  ) => DecisionTreeClassifierImpl(
    data,
    getGreedyDecisionTreeDependencies(minError, minSamplesCount),
  );
}
