import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_impl.dart';

class InterceptPreprocessorFactoryImpl implements InterceptPreprocessorFactory {
  const InterceptPreprocessorFactoryImpl();

  @override
  InterceptPreprocessor<T> create<T>({double scale}) => InterceptPreprocessorImpl(interceptScale: scale);
}
