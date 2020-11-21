import 'package:inject/inject.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_injector.inject.g.dart' as $G;
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_module.dart';

@Injector([DecisionTreeClassifierModule])
abstract class DecisionTreeClassifierInjector {
  static $G.DecisionTreeClassifierInjector$Injector fromModule(
      DecisionTreeClassifierModule module,
  ) => $G.DecisionTreeClassifierInjector$Injector.fromModule(module);

  @provide
  DecisionTreeClassifierFactory getClassifierFactory();
}
