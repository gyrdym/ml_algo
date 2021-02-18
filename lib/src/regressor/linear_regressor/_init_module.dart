import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory_impl.dart';

GetIt initLinearRegressorModule() {
  initCommonModule();

  return linearRegressorModule
    ..registerSingletonIf<LinearRegressorFactory>(
        const LinearRegressorFactoryImpl());
}
