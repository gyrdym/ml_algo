import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_linalg/dtype.dart';

class DecisionTreeClassifierFactoryImpl implements
    DecisionTreeClassifierFactory {

  const DecisionTreeClassifierFactoryImpl();

  @override
  DecisionTreeClassifier create(
      TreeSolver solver,
      String targetName,
      DType dtype,
  ) => DecisionTreeClassifierImpl(solver, targetName, dtype);
}
