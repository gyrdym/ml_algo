import 'package:inject/inject.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor_injector.inject.g.dart' as $G;
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor_module.dart';

@Injector([ModelAssessorModule])
abstract class ModelAssessorInjector {
  static $G.ModelAssessorInjector$Injector fromModule(
      ModelAssessorModule module,
  ) => $G.ModelAssessorInjector$Injector.fromModule(module);

  @provide
  ClassifierAssessor getClassifierAssessor();
}
