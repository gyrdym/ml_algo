import 'package:ml_algo/src/services/encoder_factory/encoder_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

class EncoderFactoryImpl implements EncoderFactory {
  const EncoderFactoryImpl();

  @override
  Encoder createOneHot(DataFrame fittingData, {
    Iterable<int> featureIds,
    Iterable<String> featureNames,
    String headerPrefix,
    String headerPostfix,
    UnknownValueHandlingType unknownValueHandlingType =
        UnknownValueHandlingType.ignore,
  }) => Encoder.oneHot(fittingData,
    featureNames: featureNames,
    featureIds: featureIds,
    headerPrefix: headerPrefix,
    headerPostfix: headerPostfix,
    unknownValueHandlingType: unknownValueHandlingType,
  );
}
