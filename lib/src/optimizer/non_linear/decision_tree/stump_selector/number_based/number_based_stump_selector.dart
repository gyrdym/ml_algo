import 'package:ml_linalg/matrix.dart';

abstract class NumberBasedStumpSelector {
  List<Matrix> select(Matrix observations, int selectedColumnIdx);
}
