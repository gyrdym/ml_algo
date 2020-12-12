import 'package:injector/injector.dart';

extension InjectorExtension on Injector {
  /// Registers a dependency only if it doesn't exist
  void registerSingletonIf<T>(Builder<T> builder, {
    bool override = false,
    String dependencyName = '',
  }) {
    if (exists<T>(dependencyName: dependencyName)) {
      return;
    }

    registerSingleton<T>(
      builder,
      override: override,
      dependencyName: dependencyName,
    );
  }
}
