import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/mixin/asessable_classifier_mixin.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl with AssessableClassifierMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._solver);

  final DecisionTreeSolver _solver;

  @override
  Matrix get classLabels => null;

  @override
  Matrix get coefficientsByClasses => null;

  @override
  Matrix predictClasses(Matrix features) {
    final predictedLabels = features.rows.map(_solver.getLabelForSample);

    if (predictedLabels.isEmpty) {
      return Matrix.fromColumns([]);
    }

    final _isOutcomeNominal = predictedLabels.first.nominalValue != null;

    if (_isOutcomeNominal) {
      return Matrix.fromColumns(
          predictedLabels.map((label) => label.nominalValue).toList());
    }

    return Matrix.fromColumns([
      Vector.fromList(predictedLabels.map((label) => label.numericalValue)
          .toList()),
    ]);
  }

  @override
  Matrix predictProbabilities(Matrix features) => Matrix.fromColumns([
    Vector.fromList(
        features.rows
            .map(_solver.getLabelForSample)
            .map((label) => label.probability)
            .toList(growable: false),
    ),
  ]);
}
