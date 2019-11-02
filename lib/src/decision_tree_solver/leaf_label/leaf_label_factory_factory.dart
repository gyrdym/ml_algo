import 'package:ml_algo/src/decision_tree_solver/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label/leaf_label_factory_type.dart';

abstract class TreeLeafLabelFactoryFactory {
  TreeLeafLabelFactory createByType(
      TreeLeafLabelFactoryType type);
}
