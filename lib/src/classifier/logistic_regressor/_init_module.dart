import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/link_function/logit/float64_inverse_logit_function.dart';

void initLogisticRegressorModule() {
  initCommonModule();

  logisticRegressorInjector
    ..clearAll()
    ..registerSingleton<LinkFunction>(
            () => const Float32InverseLogitLinkFunction(),
        dependencyName: float32InverseLogitLinkFunctionToken)

    ..registerSingleton<LinkFunction>(
            () => const Float64InverseLogitLinkFunction(),
        dependencyName: float64InverseLogitLinkFunctionToken);
}
