import 'dart:async';
import 'package:ml_algo/src/service/worker_manager/worker_manager.dart';
import 'package:worker_manager/src/executor.dart';

class WorkerManagerImpl implements WorkerManager {
  Completer<bool> _initCompleter;

  @override
  Executor get executor => Executor();

  @override
  Future<void> init() async {
    final isWarmedUp = (await _initCompleter?.future) ?? false;

    if (isWarmedUp) {
      return;
    }

    _initCompleter ??= Completer();

    await Executor().warmUp();
    _initCompleter.complete(true);
  }
}
