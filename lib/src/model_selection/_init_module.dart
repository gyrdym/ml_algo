import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/model_selection/_injector.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory_impl.dart';
import 'package:ml_algo/src/extensions/injector.dart';

void initModelSelectionModule() {
  initCommonModule();

  modelSelectionInjector
    ..registerSingletonIf<SplitIndicesProviderFactory>(
            () => const SplitIndicesProviderFactoryImpl());
}
