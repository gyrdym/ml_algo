import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';

extension GetItExtension on GetIt {
  /// Registers a dependency only if it isn't registered yet
  void registerSingletonIf<T>(T instance, {
    String instanceName = '',
  }) {
    if (isRegistered<T>(
      instance: instance,
      instanceName: instanceName,
    )) {
      return;
    }

    if (T is MetricFactory) {
      print('MetricFactory');
      print(instance);
    }

    registerSingleton<T>(
      instance,
      instanceName: instanceName,
    );
  }
}
