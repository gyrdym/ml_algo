import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/mixin/asessable_classifier_mixin.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

class DecisionTreeClassifierImpl with AssessableClassifierMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(DataSet data, Injector dependencies) :
        _solver = DecisionTreeSolver(
            data.toMatrix(),
            data.columnRanges,
            data.outcomeRange,
            data.rangeToEncoded,
            dependencies.getDependency<LeafDetector>(),
            dependencies.getDependency<DecisionTreeLeafLabelFactory>(),
            dependencies.getDependency<SplitSelector>(),
        ),
        _isOutcomeNominal = data.rangeToEncoded.containsKey(data.outcomeRange);

  final DecisionTreeSolver _solver;
  final bool _isOutcomeNominal;

  @override
  Matrix get classLabels => null;

  @override
  Matrix get coefficientsByClasses => null;

  @override
  Matrix predictClasses(Matrix features) {
    final predictedLabels = features.rows.map(_solver.getLeafLabelBySample);

    if (_isOutcomeNominal) {
      return Matrix.fromColumns(
          predictedLabels.map((label) => label.nominalValue).toList());
    }

    return Matrix.fromColumns([
      Vector.fromList(
          predictedLabels
              .map((label) => label.numericalValue)
              .toList(),
      ),
    ]);
  }

  @override
  Matrix predictProbabilities(Matrix features) => null;
}
