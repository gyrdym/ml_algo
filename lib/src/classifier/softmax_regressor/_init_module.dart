import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/softmax_link_function.dart';

GetIt initSoftmaxRegressorModule() {
  initCommonModule();

  return softmaxRegressorModule
    ..registerSingletonIf<LinkFunction>(const SoftmaxLinkFunction())

    ..registerSingletonIf<SoftmaxRegressorFactory>(
        SoftmaxRegressorFactoryImpl(
          softmaxRegressorModule.get<LinkFunction>(),
        ));
}
