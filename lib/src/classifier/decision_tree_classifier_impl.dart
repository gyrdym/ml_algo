import 'package:ml_algo/src/classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_solver.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class DecisionTreeClassifierImpl with AssessablePredictorMixin
    implements DecisionTreeClassifier {

  DecisionTreeClassifierImpl(this._solver, String className, this._dtype)
      : classNames = [className];

  final DType _dtype;

  final DecisionTreeSolver _solver;

  @override
  final List<String> classNames;

  @override
  DataFrame predict(DataFrame features) {
    final predictedLabels = features
        .toMatrix(_dtype)
        .rows
        .map(_solver.getLabelForSample);

    if (predictedLabels.isEmpty) {
      return DataFrame([<num>[]]);
    }

    final outcomeList = predictedLabels
        .map((label) => label.value)
        .toList(growable: false);
    final outcomeVector = Vector.fromList(outcomeList, dtype: _dtype);

    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector], dtype: _dtype),
      header: classNames,
    );
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    final probabilities = Matrix.fromColumns([
      Vector.fromList(
        features
            .toMatrix(_dtype)
            .rows
            .map(_solver.getLabelForSample)
            .map((label) => label.probability)
            .toList(growable: false),
        dtype: _dtype,
      ),
    ], dtype: _dtype);

    return DataFrame.fromMatrix(
      probabilities,
      header: classNames,
    );
  }
}
