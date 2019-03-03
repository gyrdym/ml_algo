import 'package:ml_linalg/vector.dart';

abstract class LabelsProcessor {
  Vector makeLabelsOneVsAll(Vector origLabels, double targetLabel);
}
