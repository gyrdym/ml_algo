import 'package:injector/injector.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor_injector.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor_module.dart';

Injector $modelAssessorRuntimeInjector;

Injector get modelAssessorRuntimeInjector =>
    $modelAssessorRuntimeInjector ??= Injector()
      ..registerSingleton<ClassifierAssessor>(() => ModelAssessorInjector
          .fromModule(ModelAssessorModule.getInstance())
          .getClassifierAssessor());
