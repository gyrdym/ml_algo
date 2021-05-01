import 'package:ml_algo/src/model_selection/_injector.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/split_indices_provider_factory_impl.dart';

void initModelSelectionModule() {
  if (!modelSelectionInjector.exists<SplitIndicesProviderFactory>()) {
    modelSelectionInjector.registerSingleton<SplitIndicesProviderFactory>(
        () => const SplitIndicesProviderFactoryImpl());
  }
}
