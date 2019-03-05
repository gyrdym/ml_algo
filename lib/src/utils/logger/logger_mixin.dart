import 'package:logging/logging.dart';

mixin LoggerMixin {
  Logger get logger;

  void throwException(String msg) {
    final exception = Exception(msg);
    logger.severe(msg, exception);
    throw exception;
  }
}
