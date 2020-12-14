import 'package:ml_algo/src/service/worker_manager/worker_manager.dart';
import 'package:worker_manager/src/executor.dart';

class WorkerManagerImpl implements WorkerManager {
  bool _isInitialized = false;

  @override
  Executor get executor => Executor();

  @override
  Future<void> init() async {
    if (_isInitialized) {
      return;
    }

    await Executor().warmUp();
    _isInitialized = true;
  }
}
