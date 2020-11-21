import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

abstract class EncoderFactory {
  Encoder createOneHot(DataFrame fittingData, {
    Iterable<int> featureIds,
    Iterable<String> featureNames,
    String headerPrefix,
    String headerPostfix,
    UnknownValueHandlingType unknownValueHandlingType,
  });
}
