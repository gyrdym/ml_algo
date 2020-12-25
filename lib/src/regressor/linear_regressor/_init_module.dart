import 'package:injector/injector.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory_impl.dart';

Injector initLinearRegressorModule() {
  initCommonModule();

  return linearRegressorInjector
    ..registerSingletonIf<LinearRegressorFactory>(
            () => const LinearRegressorFactoryImpl());
}
