import 'package:dart_ml/src/math/vector_interface.dart';
import 'package:dart_ml/src/enums.dart';

class DataTrainTestSplitter {
  static Map<DataCategory, List<VectorInterface>> splitMatrix(List<VectorInterface> sample, double trainRatio) {
    int ratioLength = (sample.length * trainRatio).round();

    return <DataCategory, List<VectorInterface>>{
      DataCategory.TRAIN: sample.sublist(0, ratioLength),
      DataCategory.TEST: sample.sublist(ratioLength)
    };
  }

  static Map<DataCategory, VectorInterface> splitVector(VectorInterface sample, double trainRatio) {
    int ratioLength = (sample.length * trainRatio).round();

    return <DataCategory, VectorInterface>{
      DataCategory.TRAIN: sample.fromRange(0, ratioLength),
      DataCategory.TEST: sample.fromRange(ratioLength)
    };
  }
}
