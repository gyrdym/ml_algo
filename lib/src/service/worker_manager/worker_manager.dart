import 'package:worker_manager/worker_manager.dart';

abstract class WorkerManager {
  Future<void> init();
  Executor get executor;
}
