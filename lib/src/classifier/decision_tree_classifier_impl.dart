import 'package:ml_algo/src/classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl with AssessablePredictorMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._solver, String className)
      : classNames = [className];

  final DecisionTreeSolver _solver;

  @override
  final List<String> classNames;

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
      header: classNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final probabilities = Matrix.fromColumns([
      Vector.fromList(
        features
            .toMatrix()
            .rows
            .map(_solver.getLabelForSample)
            .map((label) => label.probability)
            .toList(growable: false),
      ),
    ]);

    return DataFrame.fromMatrix(
      probabilities,
      header: classNames,
    );
  }
}
