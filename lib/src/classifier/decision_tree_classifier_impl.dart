import 'package:ml_algo/src/classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl with AssessablePredictorMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._solver, this._className);

  final DecisionTreeSolver _solver;

  final String _className;

  @override
  Matrix get classLabels => null;

  @override
  DataFrame predict(DataFrame features) {
    final predictedLabels = features
        .toMatrix()
        .rows
        .map(_solver.getLabelForSample);

    if (predictedLabels.isEmpty) {
      return DataFrame([<num>[]]);
    }

    final outcomeList = predictedLabels
        .map((label) => label.value)
        .toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector]),
      header: [_className],
    );
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
