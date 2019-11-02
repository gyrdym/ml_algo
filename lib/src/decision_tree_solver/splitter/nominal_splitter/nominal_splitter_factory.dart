import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter.dart';

abstract class NominalTreeSplitterFactory {
  NominalTreeSplitter create();
}
