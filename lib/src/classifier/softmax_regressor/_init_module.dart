import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float64_softmax_link_function.dart';

void initSoftmaxRegressorModule() {
  initCommonModule();

  softmaxRegressorInjector
    ..clearAll()
    ..registerSingleton<LinkFunction>(
            () => const Float32SoftmaxLinkFunction(),
        dependencyName: float32SoftmaxLinkFunctionToken)

    ..registerSingleton<LinkFunction>(
            () => const Float64SoftmaxLinkFunction(),
        dependencyName: float64SoftmaxLinkFunctionToken)

    ..registerSingleton<SoftmaxRegressorFactory>(
            () => const SoftmaxRegressorFactoryImpl());
}
