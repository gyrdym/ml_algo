import 'package:logging/logging.dart';

abstract class ErrorLoggerMixin {
  Logger logger;

  void throwException(String msg) {
    final exception = Exception(msg);
    logger.severe(msg, exception);
    throw exception;
  }
}