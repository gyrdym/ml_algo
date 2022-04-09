import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

abstract class DecisionTreeClassifierFactory {
  DecisionTreeClassifier create(
    DataFrame trainData,
    String targetName,
    DType dtype,
    num minError,
    int minSamplesCount,
    int maxDepth,
    TreeAssessorType assessorType,
  );

  DecisionTreeClassifier fromJson(String json);
}
