import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/logit/float64_inverse_logit_function.dart';

Injector initLogisticRegressorModule() {
  initCommonModule();

  return logisticRegressorInjector
    ..registerSingletonIf<LinkFunction>(
            () => const Float32InverseLogitLinkFunction(),
        dependencyName: float32InverseLogitLinkFunctionToken)

    ..registerSingletonIf<LinkFunction>(
            () => const Float64InverseLogitLinkFunction(),
        dependencyName: float64InverseLogitLinkFunctionToken)

    ..registerSingletonIf<LogisticRegressorFactory>(
            () => const LogisticRegressorFactoryImpl());
}
