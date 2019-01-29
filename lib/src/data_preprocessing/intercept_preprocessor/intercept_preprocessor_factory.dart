import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';

abstract class InterceptPreprocessorFactory {
  InterceptPreprocessor create(Type dtype, {double scale});
}
