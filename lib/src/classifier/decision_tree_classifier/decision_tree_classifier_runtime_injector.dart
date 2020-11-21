import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_module.dart';

Injector $decisionTreeClassifierRuntimeInjector;

Injector get decisionTreeClassifierRuntimeInjector =>
    $decisionTreeClassifierRuntimeInjector ??= Injector()
      ..registerSingleton<DecisionTreeClassifierFactory>(() =>
          DecisionTreeClassifierInjector
              .fromModule(DecisionTreeClassifierModule.getInstance())
              .getClassifierFactory(),
      );
