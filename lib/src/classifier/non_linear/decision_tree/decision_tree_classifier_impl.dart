import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_solver.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl with AssessablePredictorMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._solver);

  final DecisionTreeSolver _solver;

  @override
  Matrix get classLabels => null;

  @override
  Matrix predict(Matrix features) {
    final predictedLabels = features.rows.map(_solver.getLabelForSample);

    if (predictedLabels.isEmpty) {
      return Matrix.fromColumns([]);
    }

    final outcomeList = predictedLabels
        .map((label) => label.value)
        .toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList);

    return Matrix.fromColumns([outcomeVector]);
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
