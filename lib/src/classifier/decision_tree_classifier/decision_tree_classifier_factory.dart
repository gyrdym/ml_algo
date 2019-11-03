import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_linalg/dtype.dart';

abstract class DecisionTreeClassifierFactory {
  DecisionTreeClassifier create(
      TreeSolver solver,
      String targetName,
      DType dtype,
  );
}
