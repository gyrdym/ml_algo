import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function.dart';

Injector initLogisticRegressorModule() {
  initCommonModule();

  return logisticRegressorInjector
    ..registerSingletonIf<LinkFunction>(() => const InverseLogitLinkFunction())
    ..registerSingletonIf<LogisticRegressorFactory>(
        () => LogisticRegressorFactoryImpl(
              logisticRegressorInjector.get<LinkFunction>(),
            ));
}
