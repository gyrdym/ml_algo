import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree/greedy_classifier.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

abstract class DecisionTreeClassifier implements Classifier, Assessable {
  factory DecisionTreeClassifier.greedy(
      DataSet data,
      double minError,
      int minSamplesCount,
  ) = GreedyDecisionTreeClassifier;
}
