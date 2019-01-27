import 'package:ml_linalg/vector.dart';

abstract class LabelsProcessor<T> {
  MLVector<T> makeLabelsOneVsAll(MLVector<T> origLabels, double targetLabel);
}
